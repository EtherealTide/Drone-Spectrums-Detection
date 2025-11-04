from pathlib import Path
from typing import Any, Optional, Dict
import json


class ParameterTable:
    """
    三层结构：
    {
        "FFT": { "Length": { ... 可选 ... }, ... },          # 示例：也可直接第三层
        "DAC": {
            "DAC00": { "freq": 1.2, "phase": 0 },
            "DAC02": { "freq": 1.2, "phase": 0 }
        }
    }
    第一层：模块名（如 FFT、DAC）
    第二层：子类名（如 DAC00）
    第三层：具体参数键（如 freq、phase）
    """

    def __init__(self, filename: str = "parameters.json"):
        base_dir = Path(__file__).parent
        self.file_path = base_dir / filename
        self.parameters: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        if self.file_path.exists():
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 确保是三层 dict（宽松允许第二层直接是第三层）
            if not isinstance(data, dict):
                raise ValueError("parameters.json 顶层必须为对象")
            self.parameters = data  # 直接保存，按需逐层访问时做判断
        else:
            self.parameters = {}
            # 初次创建空文件
            self.save_parameters()

    def get_parameter(
        self,
        group: str,
        name: Optional[str] = None,
        field: Optional[str] = None,
        default: Any = None,
    ) -> Any:
        """
        - get_parameter(group) -> 返回该组整个二层字典
        - get_parameter(group, name) -> 返回该 name 的三层字典
        - get_parameter(group, name, field) -> 返回具体值
        """
        g = self.parameters.get(group)
        if g is None:
            return default
        if name is None:
            return g
        n = g.get(name) if isinstance(g, dict) else None
        if n is None:
            return default
        if field is None:
            return n
        if isinstance(n, dict):
            return n.get(field, default)
        return default

    def set_parameter(self, group: str, name: str, field: str, value: Any) -> None:
        """
        仅更新内存，不落盘。调用 save_parameters() 才会写回文件。
        """
        if group not in self.parameters or not isinstance(self.parameters[group], dict):
            self.parameters[group] = {}
        if name not in self.parameters[group] or not isinstance(
            self.parameters[group][name], dict
        ):
            self.parameters[group][name] = {}
        self.parameters[group][name][field] = value

    def save_parameters(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.parameters, f, ensure_ascii=False, indent=2)
