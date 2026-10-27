export default class Widget extends React.Component {
  render() {
    return <div className="widget">{this.props.children}</div>;
  }
}

const helper = function (x) {
  return x * 2;
};
