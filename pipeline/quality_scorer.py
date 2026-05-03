"""
质量评分模块：长度 + 残差 + 线性度 + 连续性 四维综合评分
"""

import numpy as np


class QualityScorer:
    """轨迹质量评分器（总分100）"""

    def __init__(self, config: dict):
        cfg = config["quality"]
        self.weights = {
            "length": cfg["length_weight"],
            "residual": cfg["residual_weight"],
            "linearity": cfg["linearity_weight"],
            "continuity": cfg["continuity_weight"],
        }
        self.min_frames = cfg["min_frames"]
        self.quality_threshold = cfg["quality_threshold"]
        self.max_angle_span = cfg["max_angle_span"]
        self.penalty_60 = cfg["angle_span_penalty_60"]
        self.penalty_40 = cfg["angle_span_penalty_40"]

    def score(self, trajectory: dict, residual: np.ndarray) -> dict:
        """
        计算轨迹质量分数
        返回: {
            "total": float (0-100),
            "length_score": float,
            "residual_score": float,
            "linearity_score": float,
            "continuity_score": float,
            "passed": bool
        }
        """
        traj = trajectory["trajectory"]
        points = trajectory["points"]
        total_frames = 1200

        # 1. 长度分
        traj_frames = len(traj)
        length_score = min(1.0, traj_frames / total_frames) * self.weights["length"]

        # 2. 残差分
        res_vals = self._get_residual_values(traj, residual)
        if len(res_vals) > 0:
            mean_res = np.mean(res_vals)
            if mean_res < 0.3:
                res_score = 0
            elif mean_res > 1.0:
                res_score = self.weights["residual"]
            else:
                res_score = ((mean_res - 0.3) / 0.7) * self.weights["residual"]
        else:
            res_score = 0

        # 3. 线性度分（协方差特征值比）
        if len(points) >= 3:
            cov = np.cov(points[:, 0], points[:, 1])
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            if eigenvalues[0] > 0:
                ratio = eigenvalues[1] / eigenvalues[0]
                linearity_score = (1.0 - min(1.0, ratio)) * self.weights["linearity"]
            else:
                linearity_score = self.weights["linearity"]
        else:
            linearity_score = 0

        # 4. 连续性分
        span = trajectory["t_end"] - trajectory["t_start"] + 1
        if span > 0:
            cont_ratio = trajectory["valid_count"] / span
        else:
            cont_ratio = 0
        continuity_score = min(1.0, cont_ratio) * self.weights["continuity"]

        # 总分
        total = length_score + res_score + linearity_score + continuity_score

        # 角度跨度惩罚
        angle_span = np.ptp(traj[:, 1]) if len(traj) > 1 else 0
        if angle_span > self.max_angle_span:
            total -= self.penalty_60
        elif angle_span > 40:
            total -= self.penalty_40

        total = max(0, min(100, total))

        return {
            "total": round(total, 1),
            "length_score": round(length_score, 1),
            "residual_score": round(res_score, 1),
            "linearity_score": round(linearity_score, 1),
            "continuity_score": round(continuity_score, 1),
            "passed": total >= self.quality_threshold or traj_frames >= 100,
            "angle_span": round(angle_span, 1)
        }

    def _get_residual_values(self, trajectory: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """获取轨迹各点的CFAR残差值"""
        vals = []
        for pt in trajectory:
            t = int(pt[0])
            angle = int(round(pt[1]))
            if 0 <= t < residual.shape[0] and 0 <= angle < residual.shape[1]:
                vals.append(residual[t, angle])
        return np.array(vals)
