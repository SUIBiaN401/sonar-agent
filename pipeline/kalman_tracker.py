"""
Kalman跟踪模块：卡尔曼滤波 + 间隙填充 + 信号验证
"""

import numpy as np
from typing import List, Dict, Any


class KalmanTracker:
    """多目标卡尔曼跟踪器"""

    def __init__(self, config: dict):
        cfg = config["tracker"]
        self.max_gap = cfg["max_gap"]
        self.verify_signal = cfg["verify_signal"]
        self.Q_val = cfg["process_noise"]
        self.R_val = cfg["measurement_noise"]
        self.min_confirmed = cfg["min_confirmed"]

    def track(self, cluster_result: dict, residual: np.ndarray) -> list[dict]:
        """
        对每个聚类执行卡尔曼跟踪
        返回轨迹列表，每条轨迹: {
            "id": int,
            "trajectory": np.ndarray [Tx2],  # [帧, 角度]
            "points": 原始检测点,
            "valid_count": 有效观测数,
            "t_start": int,
            "t_end": int,
            "is_confirmed": bool
        }
        """
        trajectories = []
        for label, points in cluster_result.get("cluster_points", {}).items():
            if len(points) < self.min_confirmed:
                continue
            traj = self._kalman_filter(points, residual)
            if traj is not None:
                trajectories.append(traj)
        return trajectories

    def _kalman_filter(self, points: np.ndarray, residual: np.ndarray) -> dict | None:
        """对单个聚类的点执行卡尔man滤波"""
        # 按帧排序
        order = np.argsort(points[:, 0])
        points = points[order]

        n_frames = int(points[-1, 0]) - int(points[0, 0]) + 1
        if n_frames < 3:
            return None

        # 状态: [角度, 角速度]
        x = np.array([points[0, 1], 0.0], dtype=np.float64)  # 初始状态
        P = np.eye(2, dtype=np.float64) * 10.0              # 初始协方差

        # 状态转移矩阵 F
        dt = 1.0
        F = np.array([[1, dt], [0, 1]], dtype=np.float64)
        # 过程噪声 Q
        q = self.Q_val
        Q = np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]], dtype=np.float64) * q
        # 观测矩阵 H（只观测角度）
        H = np.array([[1, 0]], dtype=np.float64)
        R = np.array([[self.R_val]], dtype=np.float64)

        trajectory = []
        valid_count = 0
        gap = 0
        t_start = int(points[0, 0])
        t_end = int(points[-1, 0])

        # 建立帧到观测的映射
        frame_obs = {}
        for pt in points:
            f = int(pt[0])
            frame_obs[f] = pt[1]

        for t in range(t_start, t_end + 1):
            # 预测
            x = F @ x
            P = F @ P @ F.T + Q

            if t in frame_obs:
                # 更新
                z = np.array([frame_obs[t]])
                y = z - H @ x
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ y
                P = (np.eye(2) - K @ H) @ P
                valid_count += 1
                gap = 0
            else:
                # 间隙填充
                gap += 1
                if gap > self.max_gap:
                    break
                # 信号验证
                if self.verify_signal:
                    pred_angle = int(round(x[0]))
                    if 0 <= pred_angle < residual.shape[1]:
                        if residual[t, pred_angle] < -1.0:
                            break  # 残差太低，终止

            trajectory.append([t, x[0]])

        trajectory = np.array(trajectory)
        is_confirmed = valid_count >= self.min_confirmed

        return {
            "trajectory": trajectory,
            "points": points,
            "valid_count": valid_count,
            "t_start": t_start,
            "t_end": t_end,
            "is_confirmed": is_confirmed
        }
