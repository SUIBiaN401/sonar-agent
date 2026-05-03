"""
任务规划器：将用户任务拆解为有序子任务 DAG
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class TaskType(Enum):
    INSPECT = "inspect"
    SINGLE = "single"
    BATCH = "batch"
    DIAGNOSE = "diagnose"
    VALIDATE = "validate"
    OPTIMIZE = "optimize"


@dataclass
class SubTask:
    """子任务定义"""
    name: str
    task_type: TaskType
    params: dict = field(default_factory=dict)
    depends_on: list = field(default_factory=list)  # 依赖的子任务名
    status: str = "pending"  # pending / running / done / failed


class TaskPlanner:
    """任务规划器：解析用户意图，生成执行计划"""

    def __init__(self):
        self.tasks: list[SubTask] = []

    def plan(self, mode: str, **kwargs) -> list[SubTask]:
        """根据模式生成子任务列表"""
        self.tasks = []

        if mode == "inspect":
            self._plan_inspect(**kwargs)
        elif mode == "single":
            self._plan_single(**kwargs)
        elif mode == "batch":
            self._plan_batch(**kwargs)
        elif mode == "diagnose":
            self._plan_diagnose(**kwargs)
        elif mode == "validate":
            self._plan_validate(**kwargs)
        elif mode == "optimize":
            self._plan_optimize(**kwargs)
        else:
            raise ValueError(f"未知模式: {mode}")

        return self.tasks

    def _plan_inspect(self, **kwargs):
        self.tasks.append(SubTask("数据检查", TaskType.INSPECT, kwargs))

    def _plan_single(self, **kwargs):
        self.tasks.append(SubTask("加载数据", TaskType.SINGLE, {**kwargs, "stage": "load"}))
        self.tasks.append(SubTask("CFAR检测", TaskType.SINGLE, {**kwargs, "stage": "cfar"}, depends_on=["加载数据"]))
        self.tasks.append(SubTask("聚类分析", TaskType.SINGLE, {**kwargs, "stage": "cluster"}, depends_on=["CFAR检测"]))
        self.tasks.append(SubTask("目标跟踪", TaskType.SINGLE, {**kwargs, "stage": "track"}, depends_on=["聚类分析"]))
        self.tasks.append(SubTask("质量评分", TaskType.SINGLE, {**kwargs, "stage": "quality"}, depends_on=["目标跟踪"]))
        self.tasks.append(SubTask("峰值验证", TaskType.SINGLE, {**kwargs, "stage": "verify"}, depends_on=["质量评分"]))
        self.tasks.append(SubTask("可视化输出", TaskType.SINGLE, {**kwargs, "stage": "visualize"}, depends_on=["峰值验证"]))
        self.tasks.append(SubTask("自我验证", TaskType.VALIDATE, kwargs, depends_on=["可视化输出"]))

    def _plan_batch(self, **kwargs):
        self.tasks.append(SubTask("数据检查", TaskType.INSPECT, kwargs))
        self.tasks.append(SubTask("批量处理", TaskType.BATCH, kwargs, depends_on=["数据检查"]))
        self.tasks.append(SubTask("跨文件验证", TaskType.VALIDATE, {**kwargs, "scope": "cross_file"}, depends_on=["批量处理"]))
        self.tasks.append(SubTask("生成报告", TaskType.VALIDATE, {**kwargs, "stage": "report"}, depends_on=["跨文件验证"]))

    def _plan_diagnose(self, **kwargs):
        self.tasks.append(SubTask("加载数据", TaskType.SINGLE, {**kwargs, "stage": "load"}))
        self.tasks.append(SubTask("CFAR检测", TaskType.SINGLE, {**kwargs, "stage": "cfar"}, depends_on=["加载数据"]))
        self.tasks.append(SubTask("深度诊断", TaskType.DIAGNOSE, kwargs, depends_on=["CFAR检测"]))

    def _plan_validate(self, **kwargs):
        self.tasks.append(SubTask("结果验证", TaskType.VALIDATE, kwargs))

    def _plan_optimize(self, **kwargs):
        self.tasks.append(SubTask("数据检查", TaskType.INSPECT, kwargs))
        self.tasks.append(SubTask("参数优化", TaskType.OPTIMIZE, kwargs, depends_on=["数据检查"]))
