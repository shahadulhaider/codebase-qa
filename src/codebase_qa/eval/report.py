"""Aggregate question outcomes into CSV and Markdown tables and PNG charts."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from codebase_qa.eval.dataset import QUESTION_TYPES
from codebase_qa.eval.metrics import LEVELS, Level, recall_at, reciprocal_rank
from codebase_qa.eval.runner import EvalResults, QuestionOutcome, RunKey

ALL = "all"

# Categorical slots in fixed order, validated for adjacent pairs on a light surface.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


@dataclass(frozen=True)
class Summary:
    run: RunKey
    level: Level
    question_type: str
    questions: int
    recall: dict[int, float]
    mrr: float
    ms_per_query: float


SummaryIndex = dict[tuple[RunKey, Level, str], Summary]


def index_summaries(rows: list[Summary]) -> SummaryIndex:
    return {(r.run, r.level, r.question_type): r for r in rows}


def summarize(results: EvalResults) -> list[Summary]:
    """Mean metrics per run, level and question type (``all`` plus each type)."""
    ks = results.config.ks
    cutoff = max(ks)
    by_group: dict[tuple[RunKey, str], list[QuestionOutcome]] = defaultdict(list)
    for o in results.outcomes:
        by_group[(o.run, ALL)].append(o)
        by_group[(o.run, o.question.type)].append(o)
    rows: list[Summary] = []
    for (run, qtype), outcomes in by_group.items():
        for level in LEVELS:
            rows.append(
                Summary(
                    run=run,
                    level=level,
                    question_type=qtype,
                    questions=len(outcomes),
                    recall={
                        k: mean(recall_at(o.ranks[level], o.n_gold[level], k) for o in outcomes)
                        for k in ks
                    },
                    mrr=mean(reciprocal_rank(o.ranks[level], cutoff) for o in outcomes),
                    ms_per_query=1000 * mean(o.seconds for o in outcomes),
                )
            )
    return rows


def write_report(
    results: EvalResults, results_dir: Path, charts_dir: Path, stamp: str
) -> list[Path]:
    """Write ``<stamp>.csv``, ``<stamp>.md``, ``<stamp>.questions.csv`` and the charts."""
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize(results)
    paths = [
        _write_csv(rows, results.config.ks, results_dir / f"{stamp}.csv"),
        _write_questions(results, results_dir / f"{stamp}.questions.csv"),
        results_dir / f"{stamp}.md",
    ]
    charts = write_charts(results, rows, charts_dir)
    paths[2].write_text(render_markdown(results, rows, charts, paths[2].parent))
    return paths + charts


def render_markdown(
    results: EvalResults, rows: list[Summary], charts: list[Path], relative_to: Path
) -> str:
    ks = results.config.ks
    cutoff = max(ks)
    lines = [f"# Retrieval eval: {results.config.name}", ""]
    lines += [
        f"Recall@k and MRR@{cutoff} over deduplicated targets. File level asks whether "
        "the right file was retrieved; symbol level asks for the right function or class.",
        "",
    ]
    index = index_summaries(rows)
    runs = list(dict.fromkeys(o.run for o in results.outcomes))
    for qs in results.corpora:
        corpus_runs = [r for r in runs if r.corpus == qs.name]
        if not corpus_runs:
            continue
        lines += [f"## {qs.name} ({qs.ref}, {qs.commit[:7]}, {len(qs.questions)} questions)", ""]
        metric_cols = [f"R@{k}" for k in ks] + ["MRR"]
        header = ["config"] + [f"{lvl} {m}" for lvl in LEVELS for m in metric_cols] + ["ms/q"]
        lines += [_md_row(header), _md_row(["---"] + ["---:"] * (len(header) - 1))]
        for run in corpus_runs:
            cells = [run.label]
            for level in LEVELS:
                s = index[(run, level, ALL)]
                cells += [f"{s.recall[k]:.2f}" for k in ks] + [f"{s.mrr:.2f}"]
            cells.append(f"{index[(run, 'symbol', ALL)].ms_per_query:.0f}")
            lines.append(_md_row(cells))
        lines += ["", f"Symbol recall@{cutoff} by question type:", ""]
        types = [t for t in QUESTION_TYPES if any(q.type == t for q in qs.questions)]
        counts = {t: sum(1 for q in qs.questions if q.type == t) for t in types}
        lines += [
            _md_row(["config"] + [f"{t} ({counts[t]})" for t in types]),
            _md_row(["---"] + ["---:"] * len(types)),
        ]
        for run in corpus_runs:
            lines.append(
                _md_row(
                    [run.label] + [f"{index[(run, 'symbol', t)].recall[cutoff]:.2f}" for t in types]
                )
            )
        builds = [b for b in results.builds if b.corpus == qs.name]
        if builds:
            lines += ["", _md_row(["model", "chunker", "chunks", "over model limit"])]
            lines.append(_md_row(["---", "---", "---:", "---:"]))
            for b in builds:
                share = "n/a" if b.truncated is None else f"{100 * b.truncated:.1f}%"
                lines.append(_md_row([b.model, b.chunker, str(b.chunks), share]))
        lines.append("")
    for chart in charts:
        lines += [f"![{chart.stem}]({_relative(chart, relative_to)})", ""]
    return "\n".join(lines)


def write_charts(results: EvalResults, rows: list[Summary], charts_dir: Path) -> list[Path]:
    """Recall@max(k) per config, one panel per corpus: by level, and by question type."""
    import matplotlib

    matplotlib.use("Agg")
    charts_dir.mkdir(parents=True, exist_ok=True)
    cutoff = max(results.config.ks)
    index = index_summaries(rows)
    runs = list(dict.fromkeys(o.run for o in results.outcomes))
    corpora = [qs.name for qs in results.corpora if any(r.corpus == qs.name for r in runs)]
    name = results.config.name
    by_level: list[tuple[str, Level, str]] = [
        (f"{level} level", level, ALL) for level in reversed(LEVELS)
    ]
    by_type: list[tuple[str, Level, str]] = [
        (t, "symbol", t)
        for t in QUESTION_TYPES
        if any(o.question.type == t for o in results.outcomes)
    ]
    return [
        _bar_chart(
            charts_dir / f"{name}-recall.png",
            f"Recall@{cutoff}",
            corpora,
            runs,
            by_level,
            index,
            cutoff,
        ),
        _bar_chart(
            charts_dir / f"{name}-by-type.png",
            f"Symbol recall@{cutoff} by question type",
            corpora,
            runs,
            by_type,
            index,
            cutoff,
        ),
    ]


def _bar_chart(
    path: Path,
    title: str,
    corpora: list[str],
    runs: list[RunKey],
    series: list[tuple[str, Level, str]],
    index: SummaryIndex,
    cutoff: int,
) -> Path:
    import matplotlib.pyplot as plt

    labels = list(dict.fromkeys(r.label for r in runs))
    bar = 0.8 / len(series)
    height = max(2.5, 0.3 * len(labels) * len(series) + 1.2)
    fig, axes = plt.subplots(
        1,
        len(corpora),
        figsize=(5 * len(corpora) + 2, height),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    fig.patch.set_facecolor(SURFACE)
    for ax, corpus in zip(axes[0], corpora, strict=True):
        ax.set_facecolor(SURFACE)
        for i, (name, level, qtype) in enumerate(series):
            ys: list[float] = []
            values: list[float] = []
            for row, label in enumerate(labels):
                run = next((r for r in runs if r.corpus == corpus and r.label == label), None)
                if run is None:
                    continue
                ys.append(row + (i - (len(series) - 1) / 2) * bar)
                values.append(index[(run, level, qtype)].recall[cutoff])
            # The surface-coloured edge leaves a gap between adjacent bars.
            ax.barh(
                ys, values, height=bar, color=SERIES[i], label=name, edgecolor=SURFACE, linewidth=2
            )
            for y, value in zip(ys, values, strict=True):
                ax.text(
                    value + 0.01, y, f"{value:.2f}", va="center", fontsize=8, color=TEXT_SECONDARY
                )
        ax.set_title(corpus, color=TEXT_PRIMARY, fontsize=11, loc="left")
        ax.set_xlim(0, 1.08)
        ax.set_yticks(range(len(labels)), labels, color=TEXT_PRIMARY, fontsize=9)
        ax.invert_yaxis()
        ax.tick_params(axis="x", colors=TEXT_SECONDARY, labelsize=8)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles, names = axes[0][0].get_legend_handles_labels()
    if len(series) > 1:
        fig.legend(  # pyright: ignore[reportUnknownMemberType]
            handles,
            names,
            loc="outside upper right",
            ncols=len(series),
            frameon=False,
            fontsize=9,
            labelcolor=TEXT_SECONDARY,
        )
    fig.suptitle(  # pyright: ignore[reportUnknownMemberType]
        title, x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=12
    )
    fig.savefig(path, dpi=150, facecolor=SURFACE)  # pyright: ignore[reportUnknownMemberType]
    plt.close(fig)
    return path


def _write_csv(rows: list[Summary], ks: tuple[int, ...], path: Path) -> Path:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "corpus",
                "model",
                "chunker",
                "mode",
                "rerank",
                "level",
                "question_type",
                "questions",
                "metric",
                "value",
            ]
        )
        for r in rows:
            key = [r.run.corpus, r.run.model, r.run.chunker, r.run.mode, r.run.rerank]
            group = [r.level, r.question_type, r.questions]
            for k in ks:
                writer.writerow([*key, *group, f"recall@{k}", f"{r.recall[k]:.4f}"])
            writer.writerow([*key, *group, f"mrr@{max(ks)}", f"{r.mrr:.4f}"])
            writer.writerow([*key, *group, "ms_per_query", f"{r.ms_per_query:.1f}"])
    return path


def _write_questions(results: EvalResults, path: Path) -> Path:
    """One row per (run, question) with first-hit ranks and the top results, for spotting misses."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "corpus",
                "config",
                "id",
                "type",
                "question",
                "gold",
                "file_rank",
                "symbol_rank",
                "symbol_found",
                "top_results",
            ]
        )
        for o in results.outcomes:
            file_rank = min(o.ranks["file"].values(), default=None)
            symbol_rank = min(o.ranks["symbol"].values(), default=None)
            writer.writerow(
                [
                    o.run.corpus,
                    o.run.label,
                    o.question.id,
                    o.question.type,
                    o.question.text,
                    " ".join(str(g) for g in o.question.gold),
                    file_rank or "",
                    symbol_rank or "",
                    f"{len(o.ranks['symbol'])}/{o.n_gold['symbol']}",
                    " ".join(o.top),
                ]
            )
    return path


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _relative(path: Path, start: Path) -> str:
    import os

    return Path(os.path.relpath(path.resolve(), start.resolve())).as_posix()
