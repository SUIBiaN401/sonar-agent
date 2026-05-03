"""
Pipeline单元测试 + 端到端集成测试
"""

import json
import os
import sys
import pytest
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from pipeline.data_loader import DataLoader
from pipeline.cfar_detector import CFARDetector
from pipeline.cluster_analyzer import ClusterAnalyzer
from pipeline.kalman_tracker import KalmanTracker
from pipeline.quality_scorer import QualityScorer
from pipeline.peak_verifier import PeakVerifier
from validation.self_validator import SelfValidator
from validation.metrics import detection_rate, false_alarm_rate, trajectory_linearity
from tests.fixture_generator import FixtureGenerator


# ── 夹具 ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def config():
    config_path = PROJECT_DIR / "config" / "default_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fixtures(config):
    """生成测试数据"""
    gen = FixtureGenerator(str(PROJECT_DIR / "tests" / "test_data"))
    return gen.generate_all_fixtures()


# ── 数据加载测试 ─────────────────────────────────────────────────

class TestDataLoader:

    def test_load_synthetic(self, config, fixtures):
        loader = DataLoader(config)
        for fixture in fixtures:
            if fixture["n_targets"] > 0:
                data = loader.load(fixture["file"])
                assert data["data"].shape == (1200, 181), f"维度错误: {data['shape']}"
                assert data["snr_class"] in ["high", "medium", "mid_low", "low", "very_low"]
                assert data["nan_ratio"] == 0.0
                break  # 只测试第一个有目标的文件

    def test_discover(self, config):
        loader = DataLoader(config)
        test_dir = str(PROJECT_DIR / "tests" / "test_data")
        files = loader.discover_files(test_dir)
        # 至少能找到生成的测试文件（如果目录存在）
        # 注意：MAT文件名格式为 q*CBFresult.mat，合成的可能不匹配
        # 这个测试主要验证不报错
        assert isinstance(files, list)


# ── CFAR检测测试 ─────────────────────────────────────────────────

class TestCFARDetector:

    def test_detect_single_target(self, config, fixtures):
        """单目标应检测到信号"""
        cfar = CFARDetector(config)
        # 找到单目标fixture
        single = next(f for f in fixtures if f["n_targets"] == 1)
        loader = DataLoader(config)
        data = loader.load(single["file"])
        result = cfar.detect(data["data"], "medium")
        assert len(result["detections"]) > 0, "应检测到目标"
        assert result["residual"].shape == (1200, 181)

    def test_noise_only(self, config, fixtures):
        """纯噪声：CFAR 应能正常运行并返回结果"""
        cfar = CFARDetector(config)
        noise = next(f for f in fixtures if f["n_targets"] == 0)
        loader = DataLoader(config)
        data = loader.load(noise["file"])
        result = cfar.detect(data["data"], "high")
        # 验证 CFAR 正常执行，返回有效的检测结果结构
        assert "detections" in result
        assert "residual" in result
        assert "mask" in result
        assert result["residual"].shape == (1200, 181)
        # 纯噪声的检测点应少于总像素的 50%
        assert len(result["detections"]) < 1200 * 181 * 0.5, \
            f"纯噪声检测点超过50%: {len(result['detections'])}"


# ── 聚类测试 ─────────────────────────────────────────────────────

class TestClusterAnalyzer:

    def test_cluster_separate_targets(self, config):
        """两个分离的目标应被分为两个聚类"""
        clusterer = ClusterAnalyzer(config)
        # 构造两个分离的点簇
        pts1 = np.array([[i, 40] for i in range(100)])
        pts2 = np.array([[i, 120] for i in range(100)])
        pts = np.vstack([pts1, pts2])
        result = clusterer.cluster(pts)
        assert result["n_clusters"] == 2, f"应分为2个聚类，实际{result['n_clusters']}"

    def test_empty_input(self, config):
        clusterer = ClusterAnalyzer(config)
        result = clusterer.cluster(np.array([]).reshape(0, 2))
        assert result["n_clusters"] == 0


# ── Kalman跟踪测试 ──────────────────────────────────────────────

class TestKalmanTracker:

    def test_track_straight_trajectory(self, config):
        """直线轨迹应被正确跟踪"""
        # 使用 verify_signal=False 避免随机残差导致中断
        cfg = {**config, "tracker": {**config["tracker"], "verify_signal": False}}
        tracker = KalmanTracker(cfg)
        residual = np.random.normal(5, 1, (1200, 181))  # 高残差避免信号验证中断
        # 模拟聚类结果：连续点
        points = np.array([[i, 60] for i in range(0, 1200, 2)])
        cluster_result = {
            "n_clusters": 1,
            "cluster_points": {0: points}
        }
        trajectories = tracker.track(cluster_result, residual)
        assert len(trajectories) > 0
        traj = trajectories[0]
        assert traj["valid_count"] > 0
        assert traj["is_confirmed"]


# ── 质量评分测试 ─────────────────────────────────────────────────

class TestQualityScorer:

    def test_long_trajectory_high_score(self, config):
        """长轨迹应得高分"""
        scorer = QualityScorer(config)
        residual = np.random.normal(3, 1, (1200, 181))
        traj = {
            "trajectory": np.array([[i, 60 + 0.01 * i] for i in range(1200)]),
            "points": np.array([[i, 60] for i in range(0, 1200, 3)]),
            "valid_count": 400,
            "t_start": 0,
            "t_end": 1199,
            "is_confirmed": True
        }
        qs = scorer.score(traj, residual)
        assert qs["total"] > 50, f"长轨迹分数应较高: {qs['total']}"

    def test_short_trajectory_low_score(self, config):
        """短轨迹应得低分"""
        scorer = QualityScorer(config)
        residual = np.random.normal(0, 1, (1200, 181))
        traj = {
            "trajectory": np.array([[i, 60] for i in range(20)]),
            "points": np.array([[i, 60] for i in range(20)]),
            "valid_count": 20,
            "t_start": 0,
            "t_end": 19,
            "is_confirmed": False
        }
        qs = scorer.score(traj, residual)
        assert qs["total"] < 70, f"短轨迹分数应较低: {qs['total']}"


# ── 指标测试 ─────────────────────────────────────────────────────

class TestMetrics:

    def test_detection_rate(self):
        assert detection_rate(3, 3) == 1.0
        assert detection_rate(0, 3) == 0.0
        assert detection_rate(2, 3) == 2 / 3
        assert detection_rate(0, 0) == 1.0

    def test_false_alarm_rate(self):
        assert false_alarm_rate(0, 10) == 0.0
        assert false_alarm_rate(3, 10) == 0.3

    def test_linearity_straight(self):
        """直线轨迹线性度应高"""
        points = np.array([[i, 60] for i in range(100)])
        lin = trajectory_linearity(points)
        assert lin > 0.5, f"直线线性度应高: {lin}"

    def test_linearity_scattered(self):
        """散乱点线性度应低"""
        np.random.seed(42)
        points = np.random.rand(100, 2) * 100
        lin = trajectory_linearity(points)
        assert lin < 0.8, f"散乱点线性度应较低: {lin}"


# ── 端到端集成测试 ──────────────────────────────────────────────

class TestEndToEnd:

    def test_agent_initialization(self, config):
        """Agent 应能正常初始化"""
        from agent.core_agent import SonarAgent
        agent = SonarAgent(
            str(PROJECT_DIR / "config" / "default_config.json"),
            str(PROJECT_DIR / "tests" / "test_output")
        )
        assert agent is not None
        assert agent.cfar is not None
        assert agent.tracker is not None
        assert agent.self_validator is not None

    def test_agent_processes_real_file(self, config):
        """Agent 应能处理真实数据文件"""
        from agent.core_agent import SonarAgent

        # 使用真实数据文件（q6 是已知的 medium SNR 文件）
        real_file = PROJECT_DIR / ".." / "骗小米" / "data" / "训练" / "q6-CBFresult.mat"
        if not real_file.exists():
            pytest.skip("真实数据文件不存在")

        agent = SonarAgent(
            str(PROJECT_DIR / "config" / "default_config.json"),
            str(PROJECT_DIR / "tests" / "test_output")
        )
        result = agent.process_single(str(real_file))

        # 验证结果结构完整
        assert "file_name" in result
        assert "n_trajectories" in result
        assert "n_estimated" in result
        assert "validation_report" in result
        assert "processing_time" in result
        # q6 是 medium SNR，应能检测到至少 1 个目标
        assert result["n_trajectories"] >= 1, f"q6 应检测到至少 1 个目标: {result['n_trajectories']}"

    def test_batch_processing(self, config):
        """Agent 应能批量处理"""
        from agent.core_agent import SonarAgent

        agent = SonarAgent(
            str(PROJECT_DIR / "config" / "default_config.json"),
            str(PROJECT_DIR / "tests" / "test_output")
        )
        # 只处理 3 个文件以加快测试
        test_files = [
            str(PROJECT_DIR / ".." / "骗小米" / "data" / "训练" / "q2-CBFresult.mat"),
            str(PROJECT_DIR / ".." / "骗小米" / "data" / "训练" / "q6-CBFresult.mat"),
            str(PROJECT_DIR / ".." / "骗小米" / "data" / "训练" / "q21-CBFresult.mat"),
        ]
        results = []
        for f in test_files:
            if Path(f).exists():
                r = agent.process_single(f)
                results.append(r)
        assert len(results) >= 1
        # 验证每个结果都有验证报告
        for r in results:
            assert "validation_report" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
