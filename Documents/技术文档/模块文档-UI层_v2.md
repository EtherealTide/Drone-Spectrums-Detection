# UI层模块文档

**目录**: `Software/UI/`

## 1. 模块概述

基于PyQt6的现代化图形界面，采用左侧导航+右侧内容的布局，支持主题切换和实时数据可视化。

---

## 2. 整体架构

### 2.1 UI层级结构

```
Window (主窗口)
  ├─► NavigationList (左侧导航栏)
  │     ├─ Spectrum (频谱界面)
  │     ├─ Waterfall (瀑布图界面)
  │     └─ Settings (设置界面)
  │
  └─► StackedWidget (右侧内容区)
        ├─► SpectrumInterface (频谱界面)
        │     ├─ SpectrumVisualizationCard (实时曲线)
        │     └─ SpectrumConfigInterface (参数配置)
        │
        ├─► WaterfallInterface (瀑布图界面)
        │     ├─ WaterfallVisualizationCard (检测图像+统计)
        │     └─ WaterfallConfigInterface (参数配置)
        │
        └─► SettingsInterface (设置界面)
              └─ ThemeManager (主题切换)
```

### 2.2 数据流向

```
后端模块 → State (信号发射) → UI组件 (槽函数更新)
         ↑                     ↓
         └──── 用户操作 ────────┘
              (参数修改/连接请求)
```

---

## 3. 主窗口 (main_ui.py)

### 3.1 Window类

```python
class Window(QMainWindow):
    def __init__(self, dataprocessor, state, detector):
        self.data_processor = dataprocessor
        self.state = state
        self.detector = detector
        self.theme_manager = get_theme_manager()
```

**职责**：
- 管理主窗口布局
- 初始化所有子界面
- 应用主题样式

**核心方法**：

#### startInterface()
```python
def startInterface(self):
    """显示启动画面（Logo）"""
    - 显示Splash Screen
    - 等待子界面加载
    - 关闭启动画面
```

#### _build_interfaces()
```python
def _build_interfaces(self):
    """构建主界面布局"""
    - 创建导航列表（左侧220px）
    - 创建StackedWidget（右侧，自适应）
    - 添加子界面：Spectrum, Waterfall, Settings
    - 连接导航切换信号
```

#### _add_page(title, widget, icon_text)
```python
def _add_page(self, title, widget, icon_text=""):
    """添加页面到导航"""
    - 使用Emoji图标（如🏠📊⚙️）
    - 自动添加到导航列表和StackedWidget
```

**窗口尺寸**：
- 初始：1200×900
- 最小：1080×600
- 自动居中显示

---

## 4. 频谱界面 (spectrum/)

### 4.1 SpectrumInterface

**文件**: `UI/spectrum/home.py`

```python
class SpectrumInterface(QWidget):
    """频谱界面：实时曲线+参数配置"""
    - Left: SpectrumVisualizationCard (3份)
    - Right: SpectrumConfigInterface (2份)
    - 使用QSplitter可调节分割比例
```

### 4.2 SpectrumVisualizationCard

**文件**: `UI/spectrum/visualization.py`

**职责**：实时频谱曲线显示

**核心组件**：
```python
self.spectrum_chart = QChart()  # QtCharts图表
self.spectrum_series = QLineSeries()  # 折线数据
self.axis_x = QValueAxis()  # X轴（频率MHz）
self.axis_y = QValueAxis()  # Y轴（功率）
```

**更新机制**：
```python
def start_update(self):
    self.update_timer.start(30)  # 33 FPS

def update_visualization(self):
    spectrum = data_processor.get_latest_spectrum()
    freq_points = np.linspace(0, sample_rate/1e6, fft_length*20)
    points = [QPointF(f, p) for f, p in zip(freq_points, spectrum)]
    self.spectrum_series.replace(points)
    
    # 自动调整Y轴范围
    self.axis_y.setRange(min_power, max_power)
```

**配置参数**：
- X轴范围：`spectrum_left_freq` ~ `spectrum_right_freq` (MHz)
- Y轴：自动缩放到数据范围

### 4.3 SpectrumConfigInterface

**文件**: `UI/spectrum/config_interface.py`

**职责**：参数配置树形界面

**树形结构**：
```
System Status
  └─ Connection (连接开关)
Receiver
  ├─ FFT_Length (下拉选择)
  ├─ Decimation_factor (下拉选择)
  ├─ Centre_frequency(MHz) (文本输入)
  └─ SPAN(MHz) (文本输入)
UI_Spectrum
  ├─ spectrum_left_freq(MHz)
  └─ spectrum_right_freq(MHz)
Detection
  ├─ conf_threshold
  ├─ iou_threshold
  └─ image_size (下拉选择)
```

**交互流程**：
```python
1. 用户修改参数输入框/下拉框
2. 点击"Set"按钮
3. 发射信号 parameter_change_request.emit(group, name, value)
4. 主程序接收信号 → 更新State → 触发后端更新
5. State发射parameters_changed → 更新UI显示的当前值
```

**信号定义**：
```python
connection_request = pyqtSignal(bool)  # 连接请求
parameter_change_request = pyqtSignal(str, str, object)  # 参数修改
```

---

## 5. 瀑布图界面 (waterfall/)

### 5.1 WaterfallInterface

**文件**: `UI/waterfall/waterfall_ui.py`

```python
class WaterfallInterface(QWidget):
    """瀑布图界面：检测结果+统计+配置"""
    - Left: WaterfallVisualizationCard (3份)
    - Right: WaterfallConfigInterface (2份)
```

### 5.2 WaterfallVisualizationCard

**文件**: `UI/waterfall/visualization.py`

**职责**：显示检测图像和统计信息

**核心组件**：
```python
# 图像显示区（上，3/4高度）
self.result_label = QLabel()
  - 显示detector.get_detection_image()（带检测框）
  - 若无检测器，显示data_processor.get_waterfall_image()
  - 自动缩放，保持纵横比

# 统计信息区（下，1/4高度）
self.stats_label = BodyLabel()
  - HTML表格格式
  - 3列布局
```

**更新机制**：
```python
def start_update(self):
    self.update_timer.start(25)  # 40 FPS

def update_visualization(self):
    # 1. 获取检测图像
    detection_image = detector.get_detection_image()
    if detection_image is None:
        detection_image = data_processor.get_waterfall_image()
    
    # 2. 转置显示（data_process保持转置为detector使用）
    if detection_image.ndim == 3:
        detection_image = np.transpose(detection_image, (1, 0, 2))
    
    # 3. 转换为QPixmap显示
    self._update_label(self.result_label, detection_image)
    
    # 4. 更新统计信息
    self._update_stats(processor_stats, detection_stats, scanner_stats)
```

**统计信息显示**：
```html
<!-- 基本统计（3列） -->
发送帧数 | 接收帧数 | 数据处理帧数
检测帧数 | 检测FPS | 显示帧数
频率范围 | 当前扫描状态

<!-- 检测结果（多列） -->
类别1: 置信度 | 类别2: 置信度 | 类别3: 置信度
...
```

### 5.3 WaterfallConfigInterface

**文件**: `UI/waterfall/config_interface.py`

**树形结构**：
```
System Status
  └─ Connection
Receiver
  ├─ FFT_Length
  ├─ Decimation_factor
  ├─ Centre_frequency(MHz)
  └─ SPAN(MHz)
Data Process
  ├─ waterfall_height
  ├─ enable_noise_filter (Enabled/Disabled)
  ├─ noise_filter_mode (subtraction/threshold)
  └─ noise_alpha
Detection
  ├─ conf_threshold
  └─ iou_threshold
Scanner
  ├─ control_lost_threshold
  ├─ scan_bandwidth_mhz
  ├─ enable_scanning (Enabled/Disabled)
  └─ start_frequency_mhz
```

**特殊控件**：
- **开关按钮**：Connection（连接/断开）
- **下拉框**：FFT_Length, enable_noise_filter等
- **文本输入**：数值参数

---

## 6. 设置界面 (settings/)

### 6.1 SettingsInterface

**文件**: `UI/settings/settings_interface.py`

**职责**：主题切换

```python
class SettingsInterface(QWidget):
    def __init__(self, parent, theme_manager):
        self.theme_manager = theme_manager
```

**主题切换**：
```python
self.theme_manager.set_theme("dark")  # 深色主题
self.theme_manager.set_theme("light")  # 浅色主题
```

### 6.2 ThemeManager

**文件**: `UI/settings/theme_manager.py`

**职责**：全局主题管理

```python
class ThemeManager(QObject):
    paletteChanged = pyqtSignal(dict)  # 主题变更信号
    
    def __init__(self):
        self.current_theme = "dark"
        self.palette = self._get_palette(self.current_theme)
    
    def set_theme(self, theme_name):
        self.current_theme = theme_name
        self.palette = self._get_palette(theme_name)
        self.paletteChanged.emit(self.palette)
```

**调色板定义**：
```python
DARK_PALETTE = {
    "window_bg": "#1e1e1e",
    "panel_bg": "#252526",
    "card_bg": "#2d2d30",
    "stack_bg": "#252526",
    "text_primary": "#cccccc",
    "text_secondary": "#999999",
    "nav_text": "#cccccc",
    "nav_selected": "#094771",
    "nav_hover": "#2a2d2e",
    "card_border": "#3e3e42",
    "accent": "#0e639c"
}

LIGHT_PALETTE = {
    "window_bg": "#f3f3f3",
    "panel_bg": "#ffffff",
    ...
}
```

**使用方式**：
```python
# 全局获取
theme_manager = get_theme_manager()

# 监听主题变更
theme_manager.paletteChanged.connect(self.apply_palette)

# 应用样式
def apply_palette(self, palette: dict):
    self.setStyleSheet(f"""
        QWidget {{
            background-color: {palette['card_bg']};
            color: {palette['text_primary']};
        }}
    """)
```

---

## 7. 工具组件 (utils/)

### 7.1 Component

**文件**: `UI/utils/component.py`

**职责**：通用UI组件工厂

```python
class Component:
    def create_card(self, parent, title=None, height=None):
        """创建卡片容器（圆角边框）"""
        card = QWidget(parent)
        layout = QVBoxLayout(card)
        if title:
            layout.addWidget(self.create_title(card, title))
        return layout, card
    
    def create_label(self, parent, text, width, height, alignment):
        """创建标签"""
        label = QLabel(text, parent)
        if width: label.setFixedWidth(width)
        if height: label.setFixedHeight(height)
        label.setAlignment(alignment)
        return label
    
    def create_switch_button(self, parent, on_text, off_text):
        """创建开关按钮"""
        from qfluentwidgets import SwitchButton
        switch = SwitchButton(parent)
        return switch
```

### 7.2 BodyLabel

**文件**: `UI/utils/component.py`

```python
class BodyLabel(QLabel):
    """富文本标签，支持HTML格式化"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.RichText)
```

---

## 8. 信号-槽连接机制

### 8.1 State → UI更新

```python
# 主程序 (main.py)
state.connection_changed.connect(spectrum_config.on_connection_state_changed)
state.parameters_changed.connect(waterfall_config.on_parameters_updated)

# UI槽函数
def on_connection_state_changed(self, is_connected):
    self.connection_switch.setChecked(is_connected)

def on_parameters_updated(self, change_info):
    key = f"{change_info['group']}.{change_info['name']}"
    if key in self._value_labels:
        self._value_labels[key].setText(str(change_info['value']))
```

### 8.2 UI → 后端请求

```python
# 配置界面发射信号
config_interface.connection_request.connect(main_controller.handle_connection)
config_interface.parameter_change_request.connect(main_controller.handle_parameter)

# 主程序处理
def handle_connection(self, connect: bool):
    if connect:
        communication.connect(state.device_ip, state.device_port)
        data_processor.start_processing()
        detector.start_detection()
    else:
        detector.stop_detection()
        data_processor.stop_processing()
        communication.disconnect()

def handle_parameter(self, group: str, name: str, value):
    state.set_parameter(group, name, value)
    # 特殊参数需手动触发后端更新
    if name == "FFT_Length":
        data_processor.set_fft_length(value)
    elif name == "enable_noise_filter":
        data_processor.set_noise_filter_parameters(value == "Enabled")
```

---

## 9. 定时器更新策略

### 9.1 更新频率

| 组件 | 定时器间隔 | 帧率 | 更新内容 |
|------|-----------|------|---------|
| SpectrumVisualizationCard | 30ms | 33 FPS | 实时频谱曲线 |
| WaterfallVisualizationCard | 25ms | 40 FPS | 检测图像+统计 |

### 9.2 启动/停止控制

```python
# 界面切换时控制更新
def on_page_changed(self, index):
    # 停止所有定时器
    spectrum_viz.stop_update()
    waterfall_viz.stop_update()
    
    # 启动当前页面定时器
    if index == 0:  # Spectrum
        spectrum_viz.start_update()
    elif index == 1:  # Waterfall
        waterfall_viz.start_update()
```

---

## 10. 样式系统

### 10.1 QSS样式应用

```python
def apply_palette(self, palette: dict):
    self.setStyleSheet(f"""
        QWidget#SpectrumInterface {{
            background-color: {palette['window_bg']};
        }}
        QListWidget#NavigationList {{
            background-color: {palette['panel_bg']};
            border-right: 1px solid {palette['card_border']};
        }}
        QListWidget::item:selected {{
            background-color: {palette['nav_selected']};
        }}
    """)
```

### 10.2 自定义按钮样式

**文件**: `UI/utils/custom_style.py`

```python
CONFIRM_BUTTON_STYLE = """
    QPushButton {
        background-color: #0e639c;
        color: white;
        border-radius: 4px;
        padding: 5px;
    }
    QPushButton:hover {
        background-color: #1177bb;
    }
    QPushButton:pressed {
        background-color: #0d5689;
    }
"""
```

---

## 11. 典型交互流程

### 11.1 连接下位机

```
1. 用户点击Connection开关 → Checked
2. 发射信号: connection_request.emit(True)
3. 主程序接收:
   - communication.connect(ip, port)
   - data_processor.start_processing()
   - detector.start_detection()
4. 连接成功 → state.communication_thread = True
5. State发射: connection_changed.emit(True)
6. UI接收: on_connection_state_changed(True)
7. 开关保持Checked状态
```

### 11.2 修改FFT长度

```
1. 用户在下拉框选择"1024"
2. 点击"Set"按钮
3. 发射信号: parameter_change_request.emit("Receiver", "FFT_Length", 1024)
4. 主程序接收:
   - state.set_parameter("Receiver", "FFT_Length", 1024)
   - data_processor.set_fft_length(1024)
   - communication.send_command("SET_FFT_LENGTH", 1024)
5. State发射: parameters_changed.emit({...})
6. UI接收: on_parameters_updated({...})
7. 显示标签更新为"1024"
```

### 11.3 实时图像显示

```
定时器循环（25ms/次）:
1. 从detector获取检测图像
2. 转置为UI显示格式（height×width×3）
3. 转换为QImage → QPixmap
4. 缩放到Label尺寸
5. 更新Label显示
6. 更新统计信息HTML
```

---

## 12. 性能优化

### 12.1 图像渲染优化

```python
# 确保数据连续性（避免复制）
image_array = np.ascontiguousarray(image_array)

# 使用QImage.Format_RGB888（高效）
qimage = QImage(
    image_array.data,
    width, height, bytes_per_line,
    QImage.Format.Format_RGB888
)

# 平滑缩放
pixmap = QPixmap.fromImage(qimage).scaled(
    width, height,
    Qt.AspectRatioMode.KeepAspectRatio,
    Qt.TransformationMode.SmoothTransformation
)
```

### 12.2 定时器优化

```python
# 使用单次触发避免积压
def update_visualization(self):
    try:
        # 处理逻辑
        pass
    finally:
        # 确保异常时定时器不停
        pass
```

### 12.3 内存优化

```python
# QPixmap复用（避免频繁创建）
self._cached_pixmap = None

def _update_label(self, label, image):
    qimage = self._array_to_qimage(image)
    if self._cached_pixmap is None or self._cached_pixmap.size() != label.size():
        self._cached_pixmap = QPixmap.fromImage(qimage).scaled(...)
    else:
        self._cached_pixmap = QPixmap.fromImage(qimage).scaled(...)
    label.setPixmap(self._cached_pixmap)
```

---

## 13. 错误处理

### 13.1 数据获取异常

```python
def update_visualization(self):
    try:
        detection_image = detector.get_detection_image()
        if detection_image is None:
            self.result_label.setText("等待数据...")
            return
    except Exception as e:
        logger.error(f"获取数据失败: {e}", exc_info=True)
        self.result_label.setText(f"显示错误: {e}")
```

### 13.2 参数验证

```python
def update_value():
    try:
        if options:
            value = input_widget.currentText()
        else:
            value = input_widget.text()
            value = float(value)  # 类型转换
    except ValueError:
        logger.warning(f"无效参数: {value}")
        return
    
    parameter_change_request.emit(group, name, value)
```

---

## 14. 扩展接口

### 14.1 添加新页面

```python
# 1. 创建新界面类
class NewInterface(QWidget):
    def __init__(self, parent, ...):
        super().__init__(parent)
        self.setup_ui()

# 2. 在Window中添加
self._add_page("NewPage", NewInterface(...), "🎨")
```

### 14.2 添加新配置参数

```python
# 1. 在State中添加属性
@property
def new_param(self):
    return self.get_parameter("NewGroup", "new_param", default_value)

# 2. 在ConfigInterface的build_tree()中添加
new_params = [("new_param", state.new_param, None)]
for name, value, options in new_params:
    self.add_parameter(parent_item, "NewGroup", name, value, options)

# 3. 在主程序中处理
def handle_parameter(self, group, name, value):
    if name == "new_param":
        some_module.set_new_param(value)
```

---

## 15. 调试技巧

### 15.1 日志输出

```python
# UI事件日志
logger.info(f"连接请求: {checked}")
logger.info(f"参数修改: {group}.{name} = {value}")

# 性能监控
logger.debug(f"更新耗时: {elapsed:.3f}s")
```

### 15.2 样式调试

```python
# 临时修改样式测试
widget.setStyleSheet("border: 2px solid red;")

# 打印当前主题
print(theme_manager.current_theme)
print(theme_manager.palette)
```

---

## 总结

UI层通过以下机制实现高效交互：

1. **信号-槽机制**：State ↔ UI双向通信
2. **定时器更新**：30-40 FPS实时刷新
3. **主题系统**：全局样式统一管理
4. **组件复用**：Component工具类简化开发
5. **线程安全**：UI操作在主线程，后端操作在工作线程
6. **错误处理**：完善的异常捕获和用户提示
