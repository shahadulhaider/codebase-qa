"""Sample module."""

import os
from dataclasses import dataclass

DEFAULT_TIMEOUT = 5.0


def retry(attempts: int = 3):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return decorator


@dataclass
class Client:
    """HTTP client."""

    base_url: str
    timeout: float = DEFAULT_TIMEOUT

    @retry(attempts=2)
    def get(self, path: str) -> str:
        return os.path.join(self.base_url, path)

    async def close(self) -> None:
        pass

    @property
    def name(self) -> str:
        return self.base_url

    @name.setter
    def name(self, value: str) -> None:
        self.base_url = value


if __name__ == "__main__":
    print(Client("http://x").get("y"))
