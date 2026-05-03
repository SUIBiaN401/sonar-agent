"""
任务规范：结构化任务描述
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TaskSpec:
    """任务规范"""
    task_type: str  # inspect / single / batch / diagnose / validate
    priority: int = 1  # 1=高, 2=中, 3=低
    files: list = field(default_factory=list)
    data_dirs: list = field(default_factory=list)
    params_override: dict = field(default_factory=dict)
    validation_required: bool = True
    output_format: str = "json"  # json / markdown / html
    n_targets: Optional[int] = None  # 诊断模式用
