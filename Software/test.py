import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.globals import CurrentConfig

# 设置 pyecharts 使用本地资源
CurrentConfig.ONLINE_HOST = "file:///" + os.path.join(os.getcwd(), "assets", "")


def create_line_chart():
    line = (
        Line()
        .add_xaxis(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        .add_yaxis("Example Data", [820, 932, 901, 934, 1290, 1330, 1320])
        .set_global_opts(title_opts=opts.TitleOpts(title="Line Chart Example"))
    )
    # 使用绝对路径保存 HTML 文件
    temp_html_path = os.path.abspath("temp_line_chart.html")
    line.render(temp_html_path)
    return temp_html_path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECharts in PyQt6")
        self.setGeometry(100, 100, 1400, 1000)

        self.web_view = QWebEngineView(self)
        self.setCentralWidget(self.web_view)

        # 获取 HTML 文件的绝对路径
        html_path = create_line_chart()

        # 使用绝对路径加载 HTML 文件
        self.web_view.setUrl(QUrl.fromLocalFile(html_path))


if __name__ == "__main__":
    # 确保 assets 文件夹存在
    assets_dir = os.path.join(os.getcwd(), "assets")
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    # 提示用户下载 echarts.min.js
    echarts_js_path = os.path.join(assets_dir, "echarts.min.js")
    if not os.path.exists(echarts_js_path):
        print(
            "请从 https://echarts.apache.org/zh/download.html 下载 echarts.min.js 文件"
        )
        print(f"并将其放置在 {assets_dir} 文件夹中")
        sys.exit(1)

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
