"""
弱目标恢复模块：交叉区域低阈值恢复（V27创新）

在已有轨迹交叉点附近的小区域内，降低CFAR阈值进行二次检测，
独立跟踪 + 严格过滤，恢复被合并的弱目标。
"""

import numpy as np
from pipeline.kalman_tracker import KalmanTracker
from pipeline.cluster_analyzer import ClusterAnalyzer


class WeakTargetRecovery:
    """弱目标恢复器"""

    def __init__(self, config: dict):
        cfg = config["weak_recovery"]
        self.temporal_window = cfg["temporal_window"]
        self.angle_window = cfg["angle_window"]
        self.min_frames = cfg["min_recovery_frames"]
        self.min_residual = cfg["min_residual"]
        self.max_overlap = cfg["max_overlap_ratio"]
        self.tracker = KalmanTracker({**config, "tracker": {**config["tracker"], "verify_signal": False}})
        self.clusterer = ClusterAnalyzer(config)

    def recover(self, data: np.ndarray, existing_trajectories: list[dict],
                cfar_result: dict, residual: np.ndarray) -> list[dict]:
        """
        在交叉区域恢复弱目标
        """
        if len(existing_trajectories) < 2:
            return []

        # 1. 识别交叉区域
        junctions = self._find_junctions(existing_trajectories)
        if not junctions:
            return []

        recovered = []
        weak_mask = cfar_result["weak_mask"]

        for junc in junctions:
            t_center, a_center = junc["time"], junc["angle"]

            # 2. 提取局部窗口
            t_lo = max(0, t_center - self.temporal_window)
            t_hi = min(data.shape[0], t_center + self.temporal_window)
            a_lo = max(0, a_center - self.angle_window)
            a_hi = min(data.shape[1], a_center + self.angle_window)

            # 3. 局部低阈值检测
            local_weak = np.zeros_like(weak_mask)
            local_weak[t_lo:t_hi, a_lo:a_hi] = weak_mask[t_lo:t_hi, a_lo:a_hi]

            det_indices = np.argwhere(local_weak)
            if len(det_indices) < self.min_frames:
                continue

            # 4. 局部聚类 + 跟踪
            cluster_result = self.clusterer.cluster(det_indices)
            for label, points in cluster_result.get("cluster_points", {}).items():
                if len(points) < self.min_frames:
                    continue
                traj = self.tracker._kalman_filter(points, residual)
                if traj is None:
                    continue

                # 5. 严格过滤
                if not self._pass_filter(traj, existing_trajectories, residual):
                    continue

                recovered.append(traj)

        return recovered

    def _find_junctions(self, trajectories: list[dict]) -> list[dict]:
        """检测轨迹交叉点"""
        junctions = []
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                t1 = trajectories[i]["trajectory"]
                t2 = trajectories[j]["trajectory"]
                # 找时间重叠区域
                t_start = max(t1[0, 0], t2[0, 0])
                t_end = min(t1[-1, 0], t2[-1, 0])
                if t_start >= t_end:
                    continue

                # 在重叠区域内找角度最接近的点
                for t in range(int(t_start), min(int(t_end) + 1, int(t_start) + 50)):
                    a1 = self._interp_angle(t1, t)
                    a2 = self._interp_angle(t2, t)
                    if a1 is not None and a2 is not None:
                        if abs(a1 - a2) < 15:
                            junctions.append({"time": t, "angle": int((a1 + a2) / 2)})
                            break

        return junctions

    def _interp_angle(self, trajectory: np.ndarray, t: int) -> float | None:
        """在轨迹中插值获取指定帧的角度"""
        frames = trajectory[:, 0]
        if t < frames[0] or t > frames[-1]:
            return None
        idx = np.searchsorted(frames, t)
        if idx == 0:
            return trajectory[0, 1]
        if idx >= len(frames):
            return trajectory[-1, 1]
        # 线性插值
        t0, t1 = frames[idx - 1], frames[idx]
        a0, a1 = trajectory[idx - 1, 1], trajectory[idx, 1]
        if t1 == t0:
            return a0
        return a0 + (a1 - a0) * (t - t0) / (t1 - t0)

    def _pass_filter(self, traj: dict, existing: list[dict], residual: np.ndarray) -> bool:
        """严格过滤恢复的轨迹"""
        # 长度检查
        if len(traj["trajectory"]) < self.min_frames:
            return False

        # 残差检查
        res_vals = []
        for pt in traj["trajectory"]:
            t, a = int(pt[0]), int(round(pt[1]))
            if 0 <= t < residual.shape[0] and 0 <= a < residual.shape[1]:
                res_vals.append(residual[t, a])
        if len(res_vals) == 0 or np.mean(res_vals) < self.min_residual:
            return False

        # 重叠检查
        for ex in existing:
            overlap_start = max(traj["t_start"], ex["t_start"])
            overlap_end = min(traj["t_end"], ex["t_end"])
            if overlap_start < overlap_end:
                overlap_frames = overlap_end - overlap_start
                traj_span = traj["t_end"] - traj["t_start"]
                if traj_span > 0 and overlap_frames / traj_span > self.max_overlap:
                    return False

        return True
