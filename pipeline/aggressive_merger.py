"""
激进合并模块：角度<8度 + 时间gap<30 合并
"""

import numpy as np


class AggressiveMerger:
    """激进合并器：消除轨迹碎片化"""

    def __init__(self, config: dict):
        cfg = config["merge"]
        self.angle_threshold = cfg["angle_threshold"]
        self.gap_threshold = cfg["gap_threshold"]
        self.max_span = cfg["max_span"]

    def merge(self, trajectories: list[dict]) -> list[dict]:
        """
        合并角度相近、时间连续的轨迹
        """
        if len(trajectories) <= 1:
            return trajectories

        # 计算每条轨迹的平均角度
        traj_info = []
        for traj in trajectories:
            mean_angle = np.mean(traj["trajectory"][:, 1])
            traj_info.append({
                "traj": traj,
                "mean_angle": mean_angle,
                "t_start": traj["t_start"],
                "t_end": traj["t_end"]
            })

        # 按平均角度排序
        traj_info.sort(key=lambda x: x["mean_angle"])

        # 贪心合并
        merged = []
        used = set()

        for i in range(len(traj_info)):
            if i in used:
                used.add(i)
                continue
            current = traj_info[i]
            used.add(i)

            changed = True
            while changed:
                changed = False
                for j in range(len(traj_info)):
                    if j in used:
                        continue
                    candidate = traj_info[j]
                    angle_diff = abs(current["mean_angle"] - candidate["mean_angle"])
                    time_gap = min(
                        abs(current["t_end"] - candidate["t_start"]),
                        abs(candidate["t_end"] - current["t_start"])
                    )

                    if angle_diff < self.angle_threshold and time_gap < self.gap_threshold:
                        # 合并
                        merged_traj = self._merge_two(current["traj"], candidate["traj"])
                        if merged_traj is not None:
                            new_angle = np.mean(merged_traj["trajectory"][:, 1])
                            new_start = merged_traj["t_start"]
                            new_end = merged_traj["t_end"]
                            span = abs(new_angle - current["mean_angle"])
                            if span < self.max_span:
                                current = {
                                    "traj": merged_traj,
                                    "mean_angle": new_angle,
                                    "t_start": new_start,
                                    "t_end": new_end
                                }
                                used.add(j)
                                changed = True

            merged.append(current["traj"])

        return merged

    def _merge_two(self, traj1: dict, traj2: dict) -> dict | None:
        """合并两条轨迹"""
        pts1 = traj1["points"]
        pts2 = traj2["points"]
        all_points = np.vstack([pts1, pts2])

        # 按帧排序
        order = np.argsort(all_points[:, 0])
        all_points = all_points[order]

        # 合并轨迹
        t1 = traj1["trajectory"]
        t2 = traj2["trajectory"]
        all_traj = np.vstack([t1, t2])
        order = np.argsort(all_traj[:, 0])
        all_traj = all_traj[order]

        return {
            "trajectory": all_traj,
            "points": all_points,
            "valid_count": traj1.get("valid_count", 0) + traj2.get("valid_count", 0),
            "t_start": int(all_points[0, 0]),
            "t_end": int(all_points[-1, 0]),
            "is_confirmed": True
        }
