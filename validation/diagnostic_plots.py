"""
诊断图生成：验证失败时自动生成深度诊断图
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class DiagnosticPlots:
    """诊断图生成器"""

    def __init__(self, config: dict):
        self.config = config
        self.dpi = config["visualization"]["dpi"]

    def generate(self, data: dict, cfar_result: dict, peak_result: dict,
                 output_dir: str) -> list[str]:
        """
        生成诊断图，返回生成的文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        generated = []
        file_name = data["file_name"]

        # 1. CFAR残差热力图
        path = os.path.join(output_dir, f"{file_name}_residual.png")
        self._plot_residual(cfar_result["residual"], file_name, path)
        generated.append(path)

        # 2. 角度直方图 + 峰值
        path = os.path.join(output_dir, f"{file_name}_histogram.png")
        self._plot_histogram(peak_result, file_name, path)
        generated.append(path)

        # 3. 滑动窗口分析
        path = os.path.join(output_dir, f"{file_name}_sliding.png")
        self._plot_sliding(data["data"], file_name, path)
        generated.append(path)

        return generated

    def _plot_residual(self, residual, file_name, path):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.set_title(f"{file_name} - CFAR残差")
        im = ax.imshow(residual.T, aspect="auto", cmap="RdYlBu_r", origin="lower",
                       extent=[0, residual.shape[0], 0, residual.shape[1]])
        ax.set_xlabel("帧")
        ax.set_ylabel("角度")
        plt.colorbar(im, ax=ax)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_histogram(self, peak_result, file_name, path):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_title(f"{file_name} - 残差角度直方图")
        hist = peak_result.get("histogram", np.zeros(181))
        bin_centers = peak_result.get("bin_centers", np.arange(181))
        ax.bar(bin_centers, hist, width=1.0, color="steelblue", alpha=0.7)
        for angle in peak_result.get("peak_angles", []):
            ax.axvline(x=angle, color="red", linestyle="--", label=f"峰值 {angle:.0f}°")
        ax.set_xlabel("角度")
        ax.set_ylabel("计数")
        ax.legend()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_sliding(self, data, file_name, path):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_title(f"{file_name} - 滑动窗口分析")
        window = 50
        n_windows = data.shape[0] // window
        means = [np.mean(data[i*window:(i+1)*window, :]) for i in range(n_windows)]
        stds = [np.std(data[i*window:(i+1)*window, :]) for i in range(n_windows)]
        x = range(n_windows)
        ax.plot(x, means, "b-o", markersize=3, label="均值")
        ax.fill_between(x, [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)], alpha=0.2)
        ax.set_xlabel("窗口序号")
        ax.set_ylabel("强度")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
