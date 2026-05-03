"""
角度跳跃分割模块：>12度跳变分割
"""


import numpy as np


class AngleJunctionSplitter:
    """角度跳变分割器：防止不同角度目标被错误聚为一类"""

    def __init__(self, config: dict):
        self.jump_threshold = config["angle_split"]["jump_threshold"]

    def split(self, detections: np.ndarray) -> list[np.ndarray]:
        """
        按角度跳变分割检测点
        返回: 分割后的检测点数组列表
        """
        if len(detections) == 0:
            return []

        # 按帧排序
        order = np.argsort(detections[:, 0])
        detections = detections[order]

        segments = []
        current = [detections[0]]

        for i in range(1, len(detections)):
            prev_angle = detections[i - 1, 1]
            curr_angle = detections[i, 1]
            if abs(curr_angle - prev_angle) > self.jump_threshold:
                if len(current) >= 3:
                    segments.append(np.array(current))
                current = [detections[i]]
            else:
                current.append(detections[i])

        if len(current) >= 3:
            segments.append(np.array(current))

        return segments
