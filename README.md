# 🔊 声纳信号处理自主 Agent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)](tests/)

一个模块化的声纳信号处理自主 Agent，具备完整的 **CFAR检测 → DBSCAN聚类 → Kalman跟踪 → 质量评分 → 峰值验证** 流水线，以及**自动自我验证**能力。

## ✨ 特性

- 🔍 **自适应 CFAR 检测** — 5 档 SNR 自适应阈值 + 边缘屏蔽
- 🧮 **DBSCAN 聚类** — cKDTree 加速，支持角度跳跃分割
- 📈 **Kalman 跟踪** — 间隙填充 + 信号验证
- 🔄 **弱目标恢复** — 交叉区域低阈值二次检测（V27 创新）
- ✅ **自我验证** — 8 维度自动质量评估 + 诊断图生成
- 📊 **可视化** — 3×9 多子图布局 + 蓝色背景
- 🧪 **完整测试** — 16 个单元测试 + 端到端集成测试

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/你的用户名/sonar-agent.git
cd sonar-agent
pip install -r requirements.txt
```

### 使用

```bash
# 数据检查
python main.py --mode inspect --data-dir /path/to/data

# 单文件处理
python main.py --mode single --file /path/to/data/q6-CBFresult.mat

# 批量处理
python main.py --mode batch --data-dir /path/to/data

# 诊断模式（已知目标数）
python main.py --mode diagnose --file /path/to/data/q6-CBFresult.mat --n-targets 3
```

### 运行测试

```bash
python -m pytest tests/ -v
```

## 📁 项目结构

```
sonar-agent/
├── config/               # 配置层
│   └── default_config.json    # 所有参数集中管理
├── agent/                # Agent 核心
│   ├── core_agent.py     # 大脑：流程编排 + 自我验证 + 反思
│   ├── memory.py         # 记忆：FACT.md + JOURNAL.jsonl
│   └── planner.py        # 任务规划器：子任务 DAG
├── pipeline/             # 信号处理流水线（10 个模块）
│   ├── data_loader.py         # 数据加载 + SNR 分类
│   ├── cfar_detector.py       # CFAR 检测（5档自适应）
│   ├── cluster_analyzer.py    # DBSCAN 聚类
│   ├── kalman_tracker.py      # Kalman 跟踪
│   ├── quality_scorer.py      # 质量评分（4维综合）
│   ├── peak_verifier.py       # 峰值验证（残差直方图）
│   ├── angle_junction_splitter.py  # 角度跳跃分割
│   ├── aggressive_merger.py        # 激进合并
│   ├── weak_target_recovery.py     # 弱目标恢复
│   └── visualizer.py              # 可视化（3×9布局）
├── validation/           # 自我验证
│   ├── self_validator.py       # 8 维度自我验证器
│   ├── cross_file_validator.py # 跨文件一致性检查
│   ├── diagnostic_plots.py     # 诊断图生成
│   └── metrics.py              # 评估指标（纯函数）
├── tools/                # 工具
│   ├── data_inspector.py       # 数据质量检查
│   └── report_generator.py     # 报告生成（JSON/Markdown）
├── interfaces/           # 接口
│   ├── cli.py                  # 命令行接口
│   └── task_spec.py            # 任务规范
├── tests/                # 测试
│   ├── fixture_generator.py    # 合成数据生成器
│   └── test_pipeline.py        # 16 个测试用例
├── main.py               # 程序入口
├── requirements.txt      # 依赖
├── LICENSE               # MIT 许可证
└── README.md             # 本文件
```

## 🔧 处理流程

```
加载数据(MAT) → CFAR检测 → 角度跳跃分割 → DBSCAN聚类
    → Kalman跟踪 → 激进合并 → 弱目标恢复
    → 质量评分 → 峰值验证 → 可视化 → 自我验证 → 报告输出
```

## ⚙️ 核心参数

所有参数集中在 `config/default_config.json` 中，无需修改代码即可调整：

| 参数组 | 关键参数 | 说明 |
|--------|---------|------|
| CFAR | strong_thr / weak_thr | 5 档 SNR 自适应双阈值 |
| Cluster | eps=10.0, min_pts=3 | DBSCAN 聚类参数 |
| Tracker | max_gap=4 | Kalman 最大间隙帧数 |
| Quality | threshold=70 | 质量分数阈值 |
| Peak | ratio=1.5 | 峰谷比阈值 |

## 🧪 自我验证

每次处理完成后自动执行 8 维度验证：

1. **数量合理性** — 轨迹数是否在合理范围
2. **轨迹长度** — 是否 > 100 帧
3. **质量分数** — 是否 ≥ 70 分
4. **残差验证** — CFAR 残差是否 > 2dB
5. **线性度** — 轨迹协方差特征值比
6. **连续性** — 有效观测比例
7. **峰值匹配** — 轨迹角度与直方图峰值匹配
8. **自洽性** — 多维度综合评估

验证结果：`PASS` / `WARN` / `FAIL` 三级评级，失败时自动生成诊断图。

## 📊 测试覆盖

```
16 passed in 12.3s

TestDataLoader        — 数据加载、文件发现
TestCFARDetector      — 单目标检测、噪声抑制
TestClusterAnalyzer   — 分离目标聚类、空输入
TestKalmanTracker     — 直线轨迹跟踪
TestQualityScorer     — 长短轨迹评分
TestMetrics           — 检测率、虚警率、线性度
TestEndToEnd          — Agent 初始化、真实文件处理、批量处理
```

## 📦 依赖

```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pytest>=7.0.0
```

## 📄 许可证

[MIT](LICENSE)
