"""
峰值验证模块：残差角度直方图 + 峰谷比验证
"""

import numpy as np
from scipy.signal import find_peaks


class PeakVerifier:
    """基于残差角度直方图的目标数验证器"""

    def __init__(self, config: dict):
        cfg = config["peak_verify"]
        self.residual_threshold = cfg["residual_threshold"]
        self.peak_valley_ratio = cfg["peak_valley_ratio"]
        self.angle_tolerance = cfg["angle_tolerance"]
        self.max_keep_mode = cfg["max_keep_mode"]

    def verify(self, trajectories: list[dict], residual: np.ndarray, cfar_mask: np.ndarray) -> dict:
        """
        验证轨迹是否对应真实目标
        返回: {
            "n_estimated": int,
            "valid_trajectories": list,
            "peak_angles": list,
            "verification_details": list
        }
        """
        # 1. 构建残差角度直方图
        strong_mask = (residual > self.residual_threshold) & cfar_mask
        det_indices = np.argwhere(strong_mask)

        if len(det_indices) == 0:
            return {
                "n_estimated": 0,
                "valid_trajectories": [],
                "peak_angles": [],
                "verification_details": []
            }

        angles = det_indices[:, 1].astype(float)
        hist, bin_edges = np.histogram(angles, bins=181, range=(0, 181))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 2. 峰值检测
        peaks, properties = find_peaks(hist, height=3, distance=8)

        # 3. 峰谷比验证
        valid_peaks = []
        for p in peaks:
            peak_val = hist[p]
            # 找左右最近的谷值
            left_min = peak_val
            for i in range(p - 1, max(0, p - 30), -1):
                if hist[i] < left_min:
                    left_min = hist[i]
            right_min = peak_val
            for i in range(p + 1, min(len(hist), p + 31)):
                if hist[i] < right_min:
                    right_min = hist[i]

            valley = max(left_min, right_min)  # 取较高的谷值（保守估计）
            if valley > 0:
                ratio = peak_val / valley
            else:
                ratio = float("inf")

            if ratio >= self.peak_valley_ratio:
                valid_peaks.append({
                    "angle": bin_centers[p],
                    "height": int(peak_val),
                    "ratio": round(ratio, 2)
                })

        # 4. 目标数估计
        n_estimated = len(valid_peaks)
        peak_angles = [p["angle"] for p in valid_peaks]

        # 5. 轨迹与峰值匹配
        valid_trajectories = []
        details = []

        if n_estimated > 0:
            # 按质量排序
            scored_trajs = []
            for traj in trajectories:
                t_end = traj["t_end"]
                t_start = traj["t_start"]
                mean_angle = np.mean(traj["trajectory"][:, 1])
                scored_trajs.append((traj, mean_angle, traj.get("quality_score", {}).get("total", 0)))

            # 每个峰值匹配最近的轨迹
            matched = set()
            for peak in valid_peaks:
                best_traj = None
                best_dist = float("inf")
                for i, (traj, mean_angle, qs) in enumerate(scored_trajs):
                    if i in matched:
                        continue
                    dist = abs(mean_angle - peak["angle"])
                    if dist < self.angle_tolerance and dist < best_dist:
                        best_dist = dist
                        best_traj = i
                if best_traj is not None:
                    matched.add(best_traj)
                    valid_trajectories.append(scored_trajs[best_traj][0])
                    details.append({
                        "peak_angle": peak["angle"],
                        "matched_angle": scored_trajs[best_traj][1],
                        "distance": round(best_dist, 1),
                        "status": "matched"
                    })

            # 未匹配但高质量轨迹保留
            for i, (traj, mean_angle, qs) in enumerate(scored_trajs):
                if i not in matched and qs > 70:
                    valid_trajectories.append(traj)
                    details.append({
                        "peak_angle": None,
                        "matched_angle": round(mean_angle, 1),
                        "distance": None,
                        "status": "unmatched_high_quality"
                    })

            # 数量限制: max_keep = n_est
            if self.max_keep_mode == "n_est" and len(valid_trajectories) > n_estimated:
                # 按质量排序取前n_estimated个
                valid_trajectories.sort(
                    key=lambda t: t.get("quality_score", {}).get("total", 0),
                    reverse=True
                )
                valid_trajectories = valid_trajectories[:n_estimated]
        else:
            # 无峰值时保留高质量轨迹
            for traj in trajectories:
                qs = traj.get("quality_score", {}).get("total", 0)
                if qs > 70:
                    valid_trajectories.append(traj)

        return {
            "n_estimated": n_estimated,
            "valid_trajectories": valid_trajectories,
            "peak_angles": peak_angles,
            "verification_details": details,
            "histogram": hist,
            "bin_centers": bin_centers
        }
