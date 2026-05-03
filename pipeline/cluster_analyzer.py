"""
聚类分析模块：DBSCAN 聚类（使用 cKDTree 加速）
"""

import numpy as np
from scipy.spatial import cKDTree


class ClusterAnalyzer:
    """DBSCAN 聚类分析器"""

    def __init__(self, config: dict):
        cfg = config["cluster"]
        self.eps = cfg["eps"]
        self.min_pts = cfg["min_pts"]

    def cluster(self, detections: np.ndarray) -> dict:
        """
        对检测点进行DBSCAN聚类
        参数:
            detections: Nx2 [帧, 角度]
        返回:
            labels: 聚类标签数组（-1=噪声）
            n_clusters: 聚类数量
            cluster_points: dict {label: points}
        """
        if len(detections) == 0:
            return {"labels": np.array([]), "n_clusters": 0, "cluster_points": {}}

        # 构建KD树
        tree = cKDTree(detections)

        n = len(detections)
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = tree.query_ball_point(detections[i], self.eps)
            if len(neighbors) < self.min_pts:
                continue  # 标记为噪声

            # 扩展聚类
            labels[i] = cluster_id
            queue = list(neighbors)
            j = 0
            while j < len(queue):
                q = queue[j]
                if not visited[q]:
                    visited[q] = True
                    q_neighbors = tree.query_ball_point(detections[q], self.eps)
                    if len(q_neighbors) >= self.min_pts:
                        queue.extend(q_neighbors)
                if labels[q] == -1:
                    labels[q] = cluster_id
                j += 1
            cluster_id += 1

        # 整理结果
        cluster_points = {}
        for label in range(cluster_id):
            mask = labels == label
            cluster_points[label] = detections[mask]

        return {
            "labels": labels,
            "n_clusters": cluster_id,
            "cluster_points": cluster_points
        }
