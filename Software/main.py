# 用户界面（主线程）：负责与用户进行交互，包括参数调节和可视化
# PyQt6：窗口、布局、信号槽
# PyQtGraph：高性能实时绘图
import sys
from qfluentwidgets import *

# create view
view = TeachingTipView(
    icon=None,
    title="Gyro Zeppeli",
    content="""“
        触网而起的网球会落到哪一侧，谁也无法知晓。
        如果那种时刻到来，我希望「女神」是存在的。
        这样的话，不管网球落到哪一边，我都会坦然接受的吧。
    ”""",
    image="resource/Gyro.jpg",
    isClosable=False,
    tailPosition=TeachingTipTailPosition.NONE,
)

# show teaching tip
TeachingTip.make(
    view=view,
    duration=1000,
    tailPosition=TeachingTipTailPosition.NONE,
)
