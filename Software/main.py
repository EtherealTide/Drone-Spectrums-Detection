import sys
import queue
import logging
from pathlib import Path
from PyQt6.QtWidgets import QApplication

# add project path
sys.path.append(str(Path(__file__).parent))

from communication import Communication
from data_process import DataProcessor
from UI.main.main_ui import Window
from algorithms import DroneDetector
from state import State

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("drone_detection.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class DroneDetectionSystem:
    """Main orchestrator for the drone detection system."""

    def __init__(self):
        logger.info("=" * 60)
        logger.info("Initializing drone detection system...")

        self.state = State()
        logger.info("System state initialized")

        self.fft_data_queue = queue.Queue(maxsize=50)

        self.communication = Communication(self.state, self.fft_data_queue)
        logger.info("Communication layer ready")

        self.data_processor = DataProcessor(self.state)
        self.data_processor.fft_data_queue = self.fft_data_queue
        logger.info("Data processor ready")

        self.detector = DroneDetector(self.state, self.data_processor, "best.pt")
        logger.info("Detector ready")

        self.app = QApplication(sys.argv)

        self.main_window = Window(
            dataprocessor=self.data_processor, state=self.state, detector=self.detector
        )
        logger.info("UI ready")

        self.setup_connections()
        logger.info("System initialization complete")
        logger.info("=" * 60)

    def setup_connections(self):
        """Wire UI signals to handlers."""
        if hasattr(self.main_window, "spectrumInterface"):
            spectrum = self.main_window.spectrumInterface
            if hasattr(spectrum, "config_interface"):
                spectrum.config_interface.connection_request.connect(
                    self.handle_connection_request
                )
                spectrum.config_interface.parameter_change_request.connect(
                    self.handle_parameter_change_request
                )

        if hasattr(self.main_window, "waterfallInterface"):
            waterfall = self.main_window.waterfallInterface
            if hasattr(waterfall, "config_interface"):
                waterfall.config_interface.connection_request.connect(
                    self.handle_connection_request
                )
                waterfall.config_interface.parameter_change_request.connect(
                    self.handle_parameter_change_request
                )
        logger.info("Signal wiring complete")

    def handle_parameter_change_request(self, group: str, name: str, value):
        """Process parameter update requests from UI."""
        logger.info(f"Handling parameter update: {group}.{name} = {value}")

        try:
            self.state.set_parameter(group, name, value)

            if group == "Receiver" and name == "FFT_Length":
                if not self.state.communication_thread:
                    logger.warning(
                        "Communication not connected; please connect device first"
                    )
                    return
                self.communication.send_command("SET_FFT_LENGTH", value)
                self.communication.set_fft_length()
                self.data_processor.set_fft_length(value)

            if group == "UI_Spectrum":
                if hasattr(self.main_window, "spectrumInterface"):
                    self.main_window.spectrumInterface.visualization_card.update_config()

            if group == "Data_Process":
                self.data_processor.set_waterfall_parameters(
                    height=self.state.waterfall_height,
                )
                if hasattr(self.main_window, "waterfallInterface"):
                    self.main_window.waterfallInterface.visualization_card.update_config()

            if group == "Detection":
                self.detector.update_detection_parameters()
            if group == "Scanner":
                self.detector.scanning_controller.update_parameters()
            logger.info("Parameter update handled")

        except Exception as e:
            logger.error(f"Parameter update failed: {e}", exc_info=True)

    def handle_connection_request(self, should_connect):
        if should_connect:
            logger.info("Received connect request...")
            self.connect_device()
        else:
            logger.info("Received disconnect request...")
            self.disconnect_device()

    def connect_device(self):
        """Connect to device and start pipelines."""
        try:
            self.communication.connect(self.state.device_ip, self.state.device_port)

            if self.state.communication_thread:
                logger.info("Device connected")
                self.communication.send_command("SET_FFT_LENGTH", self.state.fft_length)
                self.data_processor.start_processing()
                logger.info("Data processing thread started")
                self.detector.start_detection()
                logger.info("Detection thread started")

                if hasattr(self.main_window, "waterfallInterface"):
                    viz = self.main_window.waterfallInterface
                    if hasattr(viz, "visualization_card"):
                        viz.visualization_card.start_update()
                        logger.info("Waterfall visualization started")

                if hasattr(self.main_window, "spectrumInterface"):
                    viz = self.main_window.spectrumInterface
                    if hasattr(viz, "visualization_card"):
                        viz.visualization_card.start_update()
                        logger.info("Spectrum visualization started")

                return True
            else:
                logger.error("Device connection failed")
                return False

        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
            self.state._communication_thread = False
            self.state.connection_changed.emit(False)
            return False

    def disconnect_device(self):
        """Disconnect device and stop background workers."""
        try:
            logger.info("Disconnecting device...")

            if hasattr(self.main_window, "waterfallInterface"):
                viz = self.main_window.waterfallInterface
                if hasattr(viz, "visualization_card"):
                    viz.visualization_card.stop_update()

            if hasattr(self.main_window, "spectrumInterface"):
                viz = self.main_window.spectrumInterface
                if hasattr(viz, "visualization_card"):
                    viz.visualization_card.stop_update()

            self.detector.stop_detection()
            self.data_processor.stop_processing()
            self.communication.disconnect()

            logger.info("Device disconnected")
            return True

        except Exception as e:
            logger.error(f"Disconnect error: {e}", exc_info=True)
            return False

    def run(self):
        logger.info("Launching UI...")
        self.main_window.show()
        exit_code = self.app.exec()
        self.cleanup()
        return exit_code

    def cleanup(self):
        logger.info("Cleaning up system resources...")
        self.disconnect_device()
        logger.info("System shutdown complete")


def main():
    try:
        system = DroneDetectionSystem()
        sys.exit(system.run())

    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt")
        sys.exit(0)
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
