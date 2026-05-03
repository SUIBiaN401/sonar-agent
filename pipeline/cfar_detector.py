"""
CFAR检测模块：OS-CFAR + 中值滤波背景估计 + 双阈值检测
"""

import numpy as np
from scipy.signal import medfilt2d


class CFARDetector:
    """恒虚警率检测器"""

    def __init__(self, config: dict):
        self.config = config["cfar"]
        self.snr_profiles = self.config["snr_profiles"]

    def detect(self, data: np.ndarray, snr_class: str) -> dict:
        """
        执行CFAR检测
        参数:
            data: [1200x181] 声纳数据
            snr_class: SNR档位
        返回:
            detections: Nx2 检测点坐标 [帧, 角度]
            residual: 残差矩阵
            background: 背景估计
            mask: 二值检测掩码
        """
        params = self.snr_profiles.get(snr_class, self.snr_profiles["medium"])
        ms = params["ms"]
        strong_thr = params["strong_thr"]
        weak_thr = params["weak_thr"]

        # dB转换
        data_db = 10.0 * np.log10(np.clip(data, 1e-10, None))

        # 中值滤波估计背景
        background = medfilt2d(data_db, kernel_size=[ms, 1])

        # 残差 = 信号 - 背景
        residual = data_db - background

        # 边缘屏蔽
        edge_start, edge_end = self.config.get("edge_angles_to_mask", [170, 181])
        mask = np.zeros_like(residual, dtype=bool)
        mask[:, :edge_start] = True

        # 双阈值检测
        strong_mask = (residual > strong_thr) & mask
        weak_mask = (residual > weak_thr) & mask

        # 注意：不做1D持续时间过滤
        # 原因：移动目标在不同帧可能位于不同角度，列方向不连续
        # 噪声去除由后续的DBSCAN聚类(min_pts=3)和质量评分完成

        # 收集检测点
        det_indices = np.argwhere(strong_mask)

        return {
            "detections": det_indices,       # Nx2 [frame, angle]
            "residual": residual,             # 残差矩阵
            "background": background,         # 背景估计
            "mask": strong_mask,             # 二值掩码
            "weak_mask": weak_mask,          # 弱阈值掩码
            "params_used": params
        }

    def _duration_filter(self, binary_mask: np.ndarray, min_duration: int) -> np.ndarray:
        """持续时间过滤：每列做卷积，去除短于min_duration的脉冲"""
        result = np.zeros_like(binary_mask)
        n_frames = binary_mask.shape[0]
        kernel = np.ones(min_duration)

        for col in range(binary_mask.shape[1]):
            col_data = binary_mask[:, col].astype(float)
            conv = np.convolve(col_data, kernel, mode="same")
            result[:, col] = conv >= min_duration

        return result
