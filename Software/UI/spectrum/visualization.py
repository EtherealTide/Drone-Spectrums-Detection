from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtGui import QPainter, QColor, QBrush
import numpy as np
import logging

from ..utils.component import Component
from ..settings.theme_manager import get_theme_manager
from ipc import SHM_SPECTRUM_SHAPE, SHM_SPECTRUM_DTYPE

logger = logging.getLogger(__name__)


class SpectrumVisualizationCard(QWidget):
    """Spectrum-only visualization; reads directly from shared-memory."""

    def __init__(self, parent=None, shm_spectrum=None, state=None, detector=None):
        super().__init__(parent)
        self.setObjectName("SpectrumVisualizationCard")
        self.component = Component()
        self.theme_manager = get_theme_manager()
        self.state = state
        # detector kept for signature compatibility (unused)

        self._spec_arr = None
        if shm_spectrum is not None:
            self._spec_arr = np.frombuffer(
                shm_spectrum.buf, dtype=SHM_SPECTRUM_DTYPE
            ).reshape(SHM_SPECTRUM_SHAPE)

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_visualization)

        self.setup_ui()
        self.theme_manager.paletteChanged.connect(self.apply_palette)
        self.apply_palette(self.theme_manager.palette)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        chart_layout, chart_card = self.component.create_card(self, height=300)
        chart_card.setMinimumHeight(300)
        chart_card.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.spectrum_chart = self._create_spectrum_chart()
        self.spectrum_chart_view = QChartView(self.spectrum_chart)
        self.spectrum_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        chart_layout.addWidget(self.spectrum_chart_view)

        layout.addWidget(chart_card)
        layout.setStretch(0, 1)

    def apply_palette(self, palette: dict):
        self.setStyleSheet(
            f"""
            #SpectrumVisualizationCard {{
                background-color: {palette['window_bg']};
            }}
            """
        )
        card_color = QColor(palette["card_bg"])
        plot_color = QColor(palette["stack_bg"])
        self.spectrum_chart.setBackgroundBrush(QBrush(card_color))
        self.spectrum_chart.setPlotAreaBackgroundBrush(QBrush(plot_color))
        self.spectrum_chart.setPlotAreaBackgroundVisible(True)
        self.spectrum_chart.legend().setLabelBrush(
            QBrush(QColor(palette["text_primary"]))
        )
        self.spectrum_series.setColor(QColor(palette["accent"]))
        self.axis_x.setLabelsColor(QColor(palette["text_primary"]))
        self.axis_x.setTitleBrush(QBrush(QColor(palette["text_secondary"])))
        self.axis_x.setLinePenColor(QColor(palette["text_primary"]))
        self.axis_x.setGridLineColor(QColor(palette["panel_bg"]))
        self.axis_y.setLabelsColor(QColor(palette["text_primary"]))
        self.axis_y.setTitleBrush(QBrush(QColor(palette["text_secondary"])))
        self.axis_y.setLinePenColor(QColor(palette["text_primary"]))
        self.axis_y.setGridLineColor(QColor(palette["panel_bg"]))
        self.spectrum_chart_view.setStyleSheet(
            f"background-color: {palette['card_bg']}; border-radius: 12px;"
        )

    def _create_spectrum_chart(self):
        chart = QChart()
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)

        self.spectrum_series = QLineSeries()
        self.spectrum_series.setName("功率谱")
        chart.addSeries(self.spectrum_series)

        left_freq = self.state.spectrum_left_freq
        right_freq = self.state.spectrum_right_freq

        axis_x = QValueAxis()
        axis_x.setTitleText("频率 (MHz)")
        axis_x.setRange(left_freq, right_freq)
        axis_x.setLabelFormat("%.1f")
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        self.spectrum_series.attachAxis(axis_x)
        self.axis_x = axis_x

        axis_y = QValueAxis()
        axis_y.setTitleText("功率")
        axis_y.setRange(0, 1)
        axis_y.setLabelFormat("%.4f")
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        self.spectrum_series.attachAxis(axis_y)
        self.axis_y = axis_y

        return chart

    def start_update(self):
        self.update_timer.start(30)

    def stop_update(self):
        self.update_timer.stop()

    def update_spectrum(self, spectrum_data):
        if spectrum_data is None:
            return
        max_freq = self.state.sample_rate
        freq_points = np.linspace(
            0, max_freq / 1e6, self.state.total_fft_length
        )
        points = [
            QPointF(freq, power) for freq, power in zip(freq_points, spectrum_data)
        ]
        self.spectrum_series.replace(points)

    def update_visualization(self):
        try:
            if self._spec_arr is not None and self.state:
                total_fft = self.state.total_fft_length
                total_fft = max(1, min(total_fft, SHM_SPECTRUM_SHAPE[0]))
                spectrum_data = self._spec_arr[:total_fft].copy()
                min_power = float(np.min(spectrum_data))
                max_power = float(np.max(spectrum_data))
                self.axis_y.setRange(min_power, max_power)
                self.update_spectrum(spectrum_data)
        except Exception as e:
            logger.error(f"Spectrum visualization update failed: {e}", exc_info=True)

    def update_config(self):
        left_freq = self.state.spectrum_left_freq
        right_freq = self.state.spectrum_right_freq
        self.axis_x.setRange(left_freq, right_freq)


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = SpectrumVisualizationCard()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
