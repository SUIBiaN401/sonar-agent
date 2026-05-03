"""
可视化模块：3x9多子图布局 + 蓝色背景 + CFAR点迹图
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """声纳处理结果可视化器"""

    def __init__(self, config: dict):
        cfg = config["visualization"]
        self.cmap = cfg["colormap"]
        self.layout = tuple(cfg["layout"])
        self.dpi = cfg["dpi"]
        self.figsize = tuple(cfg["figsize"])

    def plot_full(self, data: dict, cfar_result: dict, cluster_result: dict,
                  trajectories: list[dict], quality_scores: list[dict],
                  peak_result: dict, output_path: str):
        """
        生成完整的3x9可视化图
        """
        fig, axes = plt.subplots(*self.layout, figsize=self.figsize)
        fig.suptitle(f"文件: {data['file_name']} | SNR: {data['snr_class']} (std={data['db_std']})",
                     fontsize=14, fontweight="bold")

        ax = axes.flat
        residual = cfar_result["residual"]
        mask = cfar_result["mask"]

        # 1. 原始数据
        self._plot_raw(ax[0], data["data"])

        # 2. CFAR点迹图
        self._plot_cfar_detections(ax[1], residual, mask)

        # 3. DBSCAN聚类图
        self._plot_clusters(ax[2], cluster_result)

        # 4. 跟踪轨迹图
        self._plot_trajectories(ax[3], data["data"], trajectories)

        # 5. 质量分数图
        self._plot_quality(ax[4], quality_scores)

        # 6. 残差角度直方图
        self._plot_residual_hist(ax[5], peak_result)

        # 7. CFAR残差热力图
        self._plot_residual_heatmap(ax[6], residual)

        # 8. 滑动窗口分析
        self._plot_sliding_window(ax[7], data["data"])

        # 9. 最终判定图
        self._plot_final(ax[8], data["data"], peak_result, trajectories)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_raw(self, ax, data):
        ax.set_title("原始数据")
        im = ax.imshow(data.T, aspect="auto", cmap=self.cmap, origin="lower",
                       extent=[0, data.shape[0], 0, data.shape[1]])
        ax.set_xlabel("帧")
        ax.set_ylabel("角度")
        plt.colorbar(im, ax=ax, shrink=0.6)

    def _plot_cfar_detections(self, ax, residual, mask):
        ax.set_title("CFAR检测点")
        det = np.argwhere(mask)
        if len(det) > 0:
            ax.scatter(det[:, 0], det[:, 1], c=residual[det[:, 0], det[:, 1]],
                       cmap="hot", s=1, alpha=0.5)
        ax.set_xlim(0, residual.shape[0])
        ax.set_ylim(0, residual.shape[1])
        ax.set_xlabel("帧")
        ax.set_ylabel("角度")

    def _plot_clusters(self, ax, cluster_result):
        ax.set_title("DBSCAN聚类")
        colors = plt.cm.tab10(np.linspace(0, 1, max(cluster_result.get("n_clusters", 1), 1)))
        for label, points in cluster_result.get("cluster_points", {}).items():
            color = colors[label % len(colors)]
            ax.scatter(points[:, 0], points[:, 1], c=[color], s=2, label=f"C{label}")
        ax.legend(fontsize=6, loc="upper right")
        ax.set_xlabel("帧")
        ax.set_ylabel("角度")

    def _plot_trajectories(self, ax, data, trajectories):
        ax.set_title(f"跟踪轨迹 ({len(trajectories)}条)")
        ax.imshow(data.T, aspect="auto", cmap=self.cmap, origin="lower",
                  extent=[0, data.shape[0], 0, data.shape[1]], alpha=0.3)
        colors = ["red", "blue", "green", "orange", "purple"]
        for i, traj in enumerate(trajectories):
            t = traj["trajectory"]
            ax.plot(t[:, 0], t[:, 1], color=colors[i % len(colors)],
                    linewidth=1.5, label=f"T{i}")
        ax.legend(fontsize=6, loc="upper right")
        ax.set_xlabel("帧")
        ax.set_ylabel("角度")

    def _plot_quality(self, ax, quality_scores):
        ax.set_title("质量分数")
        if not quality_scores:
            ax.text(0.5, 0.5, "无轨迹", transform=ax.transAxes, ha="center")
            return
        labels = [f"T{i}" for i in range(len(quality_scores))]
        totals = [q["total"] for q in quality_scores]
        colors = ["green" if q["passed"] else "red" for q in quality_scores]
        bars = ax.bar(labels, totals, color=colors, alpha=0.7)
        ax.axhline(y=70, color="gray", linestyle="--", linewidth=0.8, label="阈值=70")
        ax.set_ylim(0, 100)
        ax.set_ylabel("分数")
        for bar, score in zip(bars, totals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{score:.0f}", ha="center", fontsize=8)
        ax.legend(fontsize=6)

    def _plot_residual_hist(self, ax, peak_result):
        ax.set_title("残差角度直方图")
        hist = peak_result.get("histogram", np.zeros(181))
        bin_centers = peak_result.get("bin_centers", np.arange(181))
        ax.bar(bin_centers, hist, width=1.0, color="steelblue", alpha=0.7)
        for angle in peak_result.get("peak_angles", []):
            ax.axvline(x=angle, color="red", linestyle="--", linewidth=0.8)
        ax.set_xlabel("角度")
        ax.set_ylabel("计数")

    def _plot_residual_heatmap(self, ax, residual):
        ax.set_title("CFAR残差热力图")
        im = ax.imshow(residual.T, aspect="auto", cmap="RdYlBu_r", origin="lower",
                       extent=[0, residual.shape[0], 0, residual.shape[1],
                               np.percentile(residual, 1), np.percentile(residual, 99)])
        ax.set_xlabel("帧")
        ax.set_ylabel("角度")
        plt.colorbar(im, ax=ax, shrink=0.6)

    def _plot_sliding_window(self, ax, data):
        ax.set_title("滑动窗口分析")
        window = 50
        n_windows = data.shape[0] // window
        means = [np.mean(data[i*window:(i+1)*window, :]) for i in range(n_windows)]
        ax.plot(range(n_windows), means, "b-o", markersize=3)
        ax.set_xlabel("窗口序号")
        ax.set_ylabel("平均强度")
        ax.grid(True, alpha=0.3)

    def _plot_final(self, ax, data, peak_result, trajectories):
        ax.set_title(f"最终判定: {peak_result.get('n_estimated', 0)}个目标")
        ax.imshow(data.T, aspect="auto", cmap=self.cmap, origin="lower",
                  extent=[0, data.shape[0], 0, data.shape[1]], alpha=0.3)
        colors = ["red", "blue", "green", "orange"]
        for i, traj in enumerate(peak_result.get("valid_trajectories", [])):
            t = traj["trajectory"]
            ax.plot(t[:, 0], t[:, 1], color=colors[i % len(colors)],
                    linewidth=2, label=f"目标{i+1}")
        for angle in peak_result.get("peak_angles", []):
            ax.axhline(y=angle, color="yellow", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=6, loc="upper right")
        ax.set_xlabel("帧")
        ax.set_ylabel("角度")
