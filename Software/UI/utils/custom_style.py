BUTTON_STYLE = """
QPushButton {
    border: none;
    padding: 5px 10px;
    font-family: 'Segoe UI';
    font-size: 16px;
    color: white;
    text-align: center;
    text-decoration: none;
    margin: 4px 2px;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #45a049;
}
QPushButton:pressed {
    background-color: #3e8e41;
}
"""
# 从上到下所有参数解释：
# border: none;               # 无边框
# padding: 5px 10px;         # 内边距，上下5px，左右10px
# font-family: 'Segoe UI';   # 字体
# font-size: 16px;           # 字体大小
# color: white;              # 字体颜色
# text-align: center;        # 文字居中
# text-decoration: none;     # 无下划线
# margin: 4px 2px;          # 外边距，上下4px，左右2px
# border-radius: 4px;       # 边框圆角4px
# 鼠标悬停时背景颜色变为#45a049
# 鼠标按下时背景颜色变为#3e8e41
ADD_BUTTON_STYLE = (
    BUTTON_STYLE
    + """
QPushButton {
    background-color: #4CAF50;
}
QPushButton:hover {
    background-color: #45a049;
}
QPushButton:pressed {
    background-color: #3e8e41;
}
"""
)
CONFIRM_BUTTON_STYLE = ADD_BUTTON_STYLE
