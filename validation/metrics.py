"""
评估指标计算：纯函数集合
"""

import numpy as np
from typing import List, Dict, Optional


def detection_rate(n_detected: int, n_true: int) -> float:
    """检测率"""
    if n_true == 0:
        return 1.0 if n_detected == 0 else 0.0
    return min(1.0, n_detected / n_true)


def false_alarm_rate(n_false: int, n_total: int) -> float:
    """虚警率"""
    if n_total == 0:
        return 0.0
    return n_false / n_total


def angle_error(traj_angles: np.ndarray, true_angles: np.ndarray) -> float:
    """角度误差（匹配后最小总误差）"""
    if len(traj_angles) == 0 or len(true_angles) == 0:
        return float("inf")

    # 贪心匹配
    errors = []
    used = set()
    for ta in traj_angles:
        best = float("inf")
        best_j = -1
        for j, tr in enumerate(true_angles):
            if j in used:
                continue
            err = abs(ta - tr)
            if err < best:
                best = err
                best_j = j
        if best_j >= 0:
            used.add(best_j)
            errors.append(best)

    return np.mean(errors) if errors else float("inf")


def trajectory_continuity(trajectory: dict) -> float:
    """轨迹连续性 = 有效观测 / 轨迹跨度"""
    span = trajectory["t_end"] - trajectory["t_start"] + 1
    if span <= 0:
        return 0.0
    return trajectory["valid_count"] / span


def trajectory_linearity(points: np.ndarray) -> float:
    """轨迹线性度 = 1 - 协方差特征值比（越直越高）"""
    if len(points) < 3:
        return 0.0
    cov = np.cov(points[:, 0], points[:, 1])
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    if eigenvalues[0] <= 0:
        return 1.0
    ratio = eigenvalues[1] / eigenvalues[0]
    return 1.0 - min(1.0, ratio)


def self_consistency_score(result: dict) -> dict:
    """
    自洽性评分（无ground truth时）
    返回多个自洽性指标
    """
    trajectories = result.get("trajectories", [])
    quality_scores = result.get("quality_scores", [])
    peak_result = result.get("peak_result", {})

    n_traj = len(trajectories)
    n_est = peak_result.get("n_estimated", 0)

    # 数量一致性：轨迹数是否与估计数一致
    count_consistent = abs(n_traj - n_est) <= 1

    # 质量一致性：所有轨迹是否都通过质量阈值
    all_passed = all(
        q.get("passed", False) for q in quality_scores
    ) if quality_scores else False

    # 长度一致性：所有真实目标应有 > 100 帧
    length_ok = all(
        len(t["trajectory"]) > 100 for t in trajectories
    ) if trajectories else False

    return {
        "count_consistent": count_consistent,
        "all_quality_passed": all_passed,
        "length_check": length_ok,
        "n_trajectories": n_traj,
        "n_estimated": n_est,
        "overall": (count_consistent + all_passed + length_ok) / 3.0
    }
