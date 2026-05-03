"""
测试数据生成器：合成已知ground truth的测试数据
"""

import numpy as np
from scipy.io import savemat
from pathlib import Path
from typing import List, Dict, Optional


class FixtureGenerator:
    """合成声纳数据生成器"""

    def __init__(self, output_dir: str = "tests/test_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_frames = 1200
        self.n_angles = 181

    def generate_single_target(self, angle: float = 60.0, snr_db: float = 20.0,
                                start_frame: int = 0, end_frame: int = 1200,
                                drift: float = 0.0, filename: str = "synthetic_single.mat") -> dict:
        """
        生成单目标合成数据
        参数:
            angle: 目标中心角度
            snr_db: 信噪比(dB)
            start_frame: 目标起始帧
            end_frame: 目标结束帧
            drift: 角度漂移(度/帧)
            filename: 输出文件名
        返回: ground truth dict
        """
        data = self._generate_noise()
        gt = {
            "n_targets": 1,
            "targets": [{"angle": angle, "start": start_frame, "end": end_frame, "drift": drift}]
        }

        for f in range(start_frame, end_frame):
            a = angle + drift * (f - start_frame)
            self._add_target(data, f, a, snr_db)

        path = str(self.output_dir / filename)
        savemat(path, {"data": data})
        gt["file"] = path
        return gt

    def generate_dual_crossing(self, angle1: float = 40.0, angle2: float = 120.0,
                                snr_db: float = 10.0, crossing_frame: int = 600,
                                filename: str = "synthetic_crossing.mat") -> dict:
        """
        生成交叉双目标合成数据
        """
        data = self._generate_noise()
        gt = {
            "n_targets": 2,
            "targets": [
                {"angle_start": angle1 - 20, "angle_end": angle1 + 20, "start": 0, "end": 1200},
                {"angle_start": angle2 + 20, "angle_end": angle2 - 20, "start": 0, "end": 1200}
            ]
        }

        for f in range(1200):
            a1 = angle1 - 20 + 40 * f / 1199
            a2 = angle2 + 20 - 40 * f / 1199
            self._add_target(data, f, a1, snr_db)
            self._add_target(data, f, a2, snr_db)

        path = str(self.output_dir / filename)
        savemat(path, {"data": data})
        gt["file"] = path
        return gt

    def generate_weak_strong(self, strong_angle: float = 50.0, weak_angle: float = 130.0,
                              strong_snr: float = 15.0, weak_snr: float = 3.0,
                              filename: str = "synthetic_weak_strong.mat") -> dict:
        """
        生成强弱目标合成数据
        """
        data = self._generate_noise()
        gt = {
            "n_targets": 2,
            "targets": [
                {"angle": strong_angle, "snr": strong_snr, "start": 0, "end": 1200},
                {"angle": weak_angle, "snr": weak_snr, "start": 200, "end": 1000}
            ]
        }

        for f in range(1200):
            self._add_target(data, f, strong_angle, strong_snr)
            if 200 <= f < 1000:
                self._add_target(data, f, weak_angle, weak_snr)

        path = str(self.output_dir / filename)
        savemat(path, {"data": data})
        gt["file"] = path
        return gt

    def generate_noise_only(self, db_std: float = 7.0,
                             filename: str = "synthetic_noise.mat") -> dict:
        """
        生成纯噪声数据（无目标）
        """
        np.random.seed(42)
        noise_db = np.random.normal(loc=40.0, scale=db_std,
                                     size=(self.n_frames, self.n_angles))
        data = 10.0 ** (noise_db / 10.0)
        path = str(self.output_dir / filename)
        savemat(path, {"data": data})
        return {"n_targets": 0, "targets": [], "file": path}

    def generate_all_fixtures(self) -> List[dict]:
        """生成所有标准测试数据"""
        fixtures = []
        fixtures.append(self.generate_single_target(snr_db=20, angle=60))
        fixtures.append(self.generate_dual_crossing(snr_db=20))
        fixtures.append(self.generate_weak_strong(strong_snr=20, weak_snr=12))
        fixtures.append(self.generate_noise_only(db_std=7.0))
        return fixtures

    def _generate_noise(self, db_std: float = 7.0) -> np.ndarray:
        """生成背景噪声（正值功率），dB域std约7dB"""
        np.random.seed(None)
        # 在 dB 域生成正态分布噪声，然后转回线性域
        noise_db = np.random.normal(loc=40.0, scale=db_std,
                                     size=(self.n_frames, self.n_angles))
        return 10.0 ** (noise_db / 10.0)

    def _add_target(self, data: np.ndarray, frame: int, angle: float, snr_db: float):
        """在指定帧和角度添加目标信号"""
        angle_idx = int(round(angle))
        if angle_idx < 0 or angle_idx >= self.n_angles:
            return
        # snr_db 作为 dB 域增量，转回线性域叠加
        linear_increment = 10.0 ** (snr_db / 10.0) - 1.0
        for da in range(-3, 4):
            a = angle_idx + da
            if 0 <= a < self.n_angles:
                weight = np.exp(-0.5 * (da / 1.5) ** 2)
                data[frame, a] *= (1.0 + linear_increment * weight)
