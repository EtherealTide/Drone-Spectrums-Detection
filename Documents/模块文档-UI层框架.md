# UI层主框架 (MainWindow)

## 1. 模块概述

UI层主框架基于 PyQt6 和 qfluentwidgets，提供现代化的Fluent Design风格界面，采用导航式多页面架构。

## 2. 核心组件

### 2.1 FluentWindow
继承自 `qfluentwidgets.FluentWindow`，提供：
- 侧边导航栏
- 多页面切换
- 主题管理

### 2.2 主要页面
```python
- HomeInterface: 主页（实时监控）
- VisualizationInterface: 可视化页面（详细图表）
- ConfigInterface: 配置页面（系统参数）
```

## 3. 技术实现

### 3.1 使用的库
- **PyQt6**: Qt6的Python绑定
- **qfluentwidgets**: Fluent Design风格组件库
- **PyQtChart**: 图表绘制

### 3.2 主窗口结构
```python
class Window(FluentWindow):
    def __init__(self, dataprocessor, state):
        self.dataprocessor = dataprocessor
        self.state = state
        self.init_navigation()
        self.init_window()
```

### 3.3 导航初始化
```python
def init_navigation(self):
    # 添加主页
    self.addSubInterface(self.homeInterface, FIF.HOME, "Home")
    
    # 添加可视化页面
    self.addSubInterface(self.visualizationInterface, FIF.CHART, "Visualization")
    
    # 添加配置页面（底部）
    self.addSubInterface(
        self.configInterface, 
        FIF.SETTING, 
        "Config", 
        NavigationItemPosition.BOTTOM
    )
```

## 4. 页面组织

### 4.1 Home页面
```
HomeInterface
├── HomeVisualizationCard (左侧)
│   ├── 功率谱图
│   ├── 瀑布图
│   └── 统计信息
└── ConfigInterface (右侧)
    └── 参数树控件
```

### 4.2 Visualization页面
```
VisualizationInterface
└── DetailedVisualizationCard
    ├── 高分辨率功率谱
    ├── 大尺寸瀑布图
    └── 更多统计信息
```

### 4.3 Config页面
```
ConfigInterface
└── 参数配置树
    ├── System Status
    ├── Receiver
    ├── DAC/ADC
    └── Algorithm
```

## 5. 数据流动

```
DataProcessor
    ↓ (定时查询)
HomeVisualizationCard
    ↓ (QPainter渲染)
QPixmap → QLabel
```

## 6. 与其他模块的接口

### 6.1 接收接口
- **DataProcessor** → 通过引用获取数据
- **State** → 监听状态变化信号

### 6.2 发出接口
- **连接请求** → DroneDetectionSystem

## 7. 主题和样式

```python
# 设置主题
setTheme(Theme.LIGHT)

# 自定义样式表
self.setStyleSheet("""
    #HomeInterface { 
        background: white; 
    }
    QSplitter::handle {
        background-color: #E0E0E0;
    }
""")
```