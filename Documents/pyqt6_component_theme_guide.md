# PyQt6 组件库与主题体系说明

## 1. 总体目标
- 替换 qfluentwidgets，基于原生 PyQt6 提供卡片、标签、开关、输入框等 Fluent 风格组件。
- 保持 `Component` 工厂 API 不变，保证上层 `home`, `visualization`, `config_interface` 等界面无需重构。
- 引入统一的主题管理器，支持浅色/深色调色板，并向全部关键部件推送配色变更。

## 2. 组件库实现

### 2.1 CardWidget
- 继承 `QFrame`，启用 `WA_StyledBackground`，通过 QSS 设置圆角、边框。
- 使用 `QGraphicsDropShadowEffect` 营造浮动卡片效果，阴影颜色来自主题。
- `Component.create_card()` 返回布局+卡片组合，自动设置统一的内边距、间距。

### 2.2 BodyLabel
- 继承 `QLabel`，设置 `Segoe UI` 字体及自动换行。
- 若调用 `setTextColor()`，使用者自定义颜色；否则监听主题调色板自动更新文字颜色。
- 支持旧代码的 `BodyLabel(parent)` 调用方式（将第一个参数解析为父级）。

### 2.3 SwitchButton
- 纯 PyQt 自绘，借助 `QPropertyAnimation` 实现平滑切换。
- 轨道、指示灯、文字颜色全部由主题调色板驱动，保持亮/暗主题一致体验。

### 2.4 输入与下拉
- `create_line_edit()`/`create_combobox()` 在构建后注册主题回调，依据调色板刷新边框、背景、文字颜色。

## 3. ThemeManager
- 位于 `Software/UI/settings/theme_manager.py`。
- `Theme` 枚举包含 `LIGHT`、`DARK`；`THEME_PALETTES` 描述窗口背景、卡片、文本、强调色、输入框等 20+ 项。
- `ThemeManager` 负责存储当前主题并发射 `themeChanged`、`paletteChanged` 信号。
- 使用 `get_theme_manager()` 获取单例，所有组件和界面在初始化时订阅调色板更新。

## 4. 主题切换入口
- `SettingsInterface`（`Software/UI/settings/settings_interface.py`）在设置页渲染“Light Theme / Dark Theme”两个按钮。
- 按钮点击调用 `ThemeManager.set_theme()`，主窗体和所有订阅者瞬时收到 `paletteChanged`，无需刷新 UI 层结构。

## 5. 主题应用路径

### 5.1 主窗体
- `Window.apply_palette()` 用调色板刷新 `QMainWindow`、导航栏、`QStackedWidget`、全局标签颜色，从根源统一色彩。

### 5.2 Home 页面
- `HomeInterface.apply_palette()` 控制背景与 `QSplitter` 句柄颜色。
- `HomeVisualizationCard.apply_palette()` 为 QChart/CardWidget/统计标签设置背景、坐标轴、折线颜色，暗色下不再出现亮色块。
- `ConfigInterface.apply_palette()` 同步树状控件、参数值标签（chip）背景，实现整体暗色适配。

### 5.3 底层组件
- `CardWidget`、`BodyLabel`、`SwitchButton`、LineEdit/ComboBox 都直接订阅主题，确保持久一致性。

## 6. 使用方式
1. 在任何界面 `from ..settings.theme_manager import get_theme_manager`，拿到单例。
2. 在 `__init__` 或 `setup_ui` 后调用 `theme_manager.paletteChanged.connect(self.apply_palette)` 并立即执行一次 `apply_palette(theme_manager.palette)`。
3. 在 `apply_palette` 中使用调色板字典（例如 `palette['card_bg']`、`palette['text_primary']`）更新自身控件。

## 7. 扩展建议
- 若新增组件，请在构造函数中订阅调色板，这样主题切换即可自动触发样式刷新。
- 如需额外主题，可在 `THEME_PALETTES` 中新增配置并扩展设置界面的按钮。

