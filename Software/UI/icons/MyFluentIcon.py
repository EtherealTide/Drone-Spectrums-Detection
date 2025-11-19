from enum import Enum
from pathlib import Path


class MyFluentIcon(Enum):
    """Lightweight helper that resolves project specific SVG icon paths."""

    CONFIG = "Config"
    VISUALIZATION = "Visualization"

    def path(self) -> str:
        ui_dir = Path(__file__).parent.parent
        return str(ui_dir / "icons" / f"{self.value}.svg")
