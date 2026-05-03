"""
数据加载模块：MAT文件读取、SNR估计、数据校验
"""

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.io import loadmat


class DataLoader:
    """声纳数据加载器"""

    # SNR分类阈值（基于dB_std）
    SNR_THRESHOLDS = {
        "high":    (0, 5),
        "medium":  (5, 7),
        "mid_low": (7, 8.5),
        "low":     (8.5, 9.5),
        "very_low": (9.5, float("inf"))
    }

    def __init__(self, config: dict):
        self.config = config

    def load(self, file_path: str) -> dict:
        """
        加载单个MAT文件
        返回: {
            "data": np.ndarray [1200x181],
            "file_name": str,
            "snr_class": str,
            "db_std": float,
            "nan_ratio": float,
            "shape": tuple
        }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        mat = loadmat(str(path))
        data = self._extract_data(mat)

        # 数据校验
        nan_ratio = np.isnan(data).sum() / data.size
        if nan_ratio > 0.5:
            raise ValueError(f"NaN比例过高: {nan_ratio:.1%}")

        # NaN处理：用列均值填充
        col_mean = np.nanmean(data, axis=0)
        for j in range(data.shape[1]):
            mask = np.isnan(data[:, j])
            if mask.any():
                data[mask, j] = col_mean[j]

        # 无穷值处理
        data = np.where(np.isinf(data), np.nanmean(data), data)

        # SNR估计：先转dB域再计算std
        data_db = 10.0 * np.log10(np.clip(data, 1e-10, None))
        db_std = float(np.nanstd(data_db))
        snr_class = self._classify_snr(db_std)

        return {
            "data": data,
            "file_name": path.stem,
            "snr_class": snr_class,
            "db_std": round(db_std, 2),
            "nan_ratio": round(nan_ratio, 4),
            "shape": data.shape
        }

    def discover_files(self, data_dir: str) -> list[str]:
        """自动发现目录中的MAT文件"""
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"目录不存在: {data_dir}")
        files = sorted(data_path.glob("q*CBFresult.mat"))
        if not files:
            # 也尝试不带连字符的格式
            files = sorted(data_path.glob("q*CBFresult*.mat"))
        return [str(f) for f in files]

    def _extract_data(self, mat: dict) -> np.ndarray:
        """从MAT字典中提取声纳数据矩阵"""
        candidates = []
        for key, val in mat.items():
            if key.startswith("__"):
                continue
            if isinstance(val, np.ndarray) and val.ndim == 2:
                candidates.append((key, val))

        if not candidates:
            raise ValueError("MAT文件中未找到二维数值数组")

        # 优先选择 (1200, 181) 形状的
        for key, val in candidates:
            if val.shape == (1200, 181):
                return val.astype(np.float64)

        # 选择最接近的
        best = max(candidates, key=lambda x: x[1].size)
        data = best[1].astype(np.float64)
        # 如果shape不对，尝试reshape
        if data.shape[0] == 181 and data.shape[1] == 1200:
            data = data.T
        return data

    def _classify_snr(self, db_std: float) -> str:
        """根据dB_std分类SNR档位"""
        for cls, (lo, hi) in self.SNR_THRESHOLDS.items():
            if lo <= db_std < hi:
                return cls
        return "very_low"
