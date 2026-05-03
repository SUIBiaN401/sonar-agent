"""
自我验证器：多维度评估输出质量
"""

import numpy as np
from typing import List, Dict, Any

from .metrics import (
    trajectory_continuity, trajectory_linearity,
    self_consistency_score
)


class SelfValidator:
    """Agent自我验证器"""

    def __init__(self, config: dict):
        cfg = config["validation"]
        self.min_frames = cfg["min_trajectory_frames"]
        self.min_residual = cfg["min_residual_db"]
        self.min_linearity = cfg["min_linearity_ratio"]
        self.min_continuity = cfg["min_continuity_ratio"]

    def validate(self, result: dict) -> dict:
        """
        对处理结果进行全面验证
        返回验证报告: {
            "overall": "PASS" | "WARN" | "FAIL",
            "checks": [...],
            "summary": str
        }
        """
        checks = []
        trajectories = result.get("valid_trajectories", result.get("trajectories", []))
        quality_scores = result.get("quality_scores", [])
        peak_result = result.get("peak_result", {})
        residual = result.get("residual", None)

        # 1. 数量合理性
        n_traj = len(trajectories)
        n_est = peak_result.get("n_estimated", 0)
        count_ok = 0 <= n_traj <= 5 and abs(n_traj - n_est) <= 2
        checks.append({
            "name": "数量合理性",
            "status": "PASS" if count_ok else "WARN",
            "detail": f"轨迹数={n_traj}, 估计数={n_est}"
        })

        # 2. 轨迹长度
        for i, traj in enumerate(trajectories):
            n_frames = len(traj.get("trajectory", []))
            length_ok = n_frames >= self.min_frames
            checks.append({
                "name": f"轨迹{i}长度",
                "status": "PASS" if length_ok else "WARN",
                "detail": f"{n_frames}帧 (阈值={self.min_frames})"
            })

        # 3. 质量分数
        for i, qs in enumerate(quality_scores):
            passed = qs.get("passed", False)
            total = qs.get("total", 0)
            checks.append({
                "name": f"轨迹{i}质量",
                "status": "PASS" if passed else "WARN",
                "detail": f"分数={total:.1f} (阈值=70)"
            })

        # 4. 残差验证
        if residual is not None:
            for i, traj in enumerate(trajectories):
                res_vals = self._get_residual(traj, residual)
                if len(res_vals) > 0:
                    mean_res = np.mean(res_vals)
                    res_ok = mean_res >= self.min_residual
                    checks.append({
                        "name": f"轨迹{i}残差",
                        "status": "PASS" if res_ok else "WARN",
                        "detail": f"平均残差={mean_res:.2f}dB (阈值={self.min_residual})"
                    })

        # 5. 线性度
        for i, traj in enumerate(trajectories):
            pts = traj.get("points", np.array([]))
            if len(pts) >= 3:
                lin = trajectory_linearity(pts)
                lin_ok = lin >= self.min_linearity
                checks.append({
                    "name": f"轨迹{i}线性度",
                    "status": "PASS" if lin_ok else "WARN",
                    "detail": f"线性度={lin:.3f} (阈值={self.min_linearity})"
                })

        # 6. 连续性
        for i, traj in enumerate(trajectories):
            cont = trajectory_continuity(traj)
            cont_ok = cont >= self.min_continuity
            checks.append({
                "name": f"轨迹{i}连续性",
                "status": "PASS" if cont_ok else "WARN",
                "detail": f"连续性={cont:.3f} (阈值={self.min_continuity})"
            })

        # 7. 峰值一致性
        peak_angles = peak_result.get("peak_angles", [])
        for i, traj in enumerate(trajectories):
            mean_angle = np.mean(traj["trajectory"][:, 1]) if len(traj["trajectory"]) > 0 else None
            if mean_angle is not None and peak_angles:
                matched = any(abs(mean_angle - pa) < 10 for pa in peak_angles)
                checks.append({
                    "name": f"轨迹{i}峰值匹配",
                    "status": "PASS" if matched else "WARN",
                    "detail": f"轨迹角度={mean_angle:.1f}, 峰值={peak_angles}"
                })

        # 8. 自洽性
        consistency = self_consistency_score(result)
        checks.append({
            "name": "自洽性",
            "status": "PASS" if consistency["overall"] >= 0.6 else "WARN",
            "detail": f"自洽分数={consistency['overall']:.2f}"
        })

        # 汇总
        n_pass = sum(1 for c in checks if c["status"] == "PASS")
        n_warn = sum(1 for c in checks if c["status"] == "WARN")
        n_fail = sum(1 for c in checks if c["status"] == "FAIL")
        total = len(checks)

        if n_fail > 0:
            overall = "FAIL"
        elif n_warn > total * 0.4:
            overall = "WARN"
        else:
            overall = "PASS"

        summary = f"验证结果: {n_pass}/{total} 通过, {n_warn} 警告, {n_fail} 失败"

        return {
            "overall": overall,
            "checks": checks,
            "summary": summary,
            "pass_rate": n_pass / total if total > 0 else 0
        }

    def _get_residual(self, traj: dict, residual: np.ndarray) -> np.ndarray:
        vals = []
        for pt in traj.get("trajectory", []):
            t, a = int(pt[0]), int(round(pt[1]))
            if 0 <= t < residual.shape[0] and 0 <= a < residual.shape[1]:
                vals.append(residual[t, a])
        return np.array(vals)
