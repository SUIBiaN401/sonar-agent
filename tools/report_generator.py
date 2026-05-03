"""
报告生成器：JSON / Markdown / HTML
"""

import json
import os
from pathlib import Path
from typing import List, Dict


class ReportGenerator:
    """报告生成器"""

    def __init__(self, config: dict):
        self.config = config

    def generate_markdown(self, results: List[dict], output_path: str):
        """生成Markdown报告"""
        lines = [
            "# 声纳信号处理报告",
            "",
            f"处理文件数: {len(results)}",
            f"总轨迹数: {sum(r.get('n_trajectories', 0) for r in results)}",
            f"平均轨迹数/文件: {sum(r.get('n_trajectories', 0) for r in results) / max(len(results), 1):.1f}",
            "",
            "## 详细结果",
            "",
            "| 文件 | SNR | 目标数 | 估计 | 峰值角度 | 验证 | 耗时(s) |",
            "|------|-----|--------|------|----------|------|---------|"
        ]

        for r in results:
            peaks = r.get("peak_angles", [])
            peak_str = ", ".join(f"{p:.0f}°" for p in peaks) if peaks else "-"
            val = r.get("validation_report", {}).get("overall", "?")
            lines.append(
                f"| {r.get('file_name', '?')} "
                f"| {r.get('snr_class', '?')} "
                f"| {r.get('n_trajectories', 0)} "
                f"| {r.get('n_estimated', 0)} "
                f"| {peak_str} "
                f"| {val} "
                f"| {r.get('processing_time', 0):.1f} |"
            )

        lines.extend(["", "## 验证详情", ""])
        for r in results:
            vr = r.get("validation_report", {})
            if vr:
                lines.append(f"### {r.get('file_name', '?')}")
                lines.append(f"- 总体: **{vr.get('overall', '?')}**")
                lines.append(f"- 通过率: {vr.get('pass_rate', 0):.0%}")
                for check in vr.get("checks", []):
                    icon = "1" if check["status"] == "PASS" else "0"
                    lines.append(f"  - {icon} {check['name']}: {check['detail']}")
                lines.append("")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def generate_json(self, results: List[dict], output_path: str):
        """生成JSON报告"""
        summary = {
            "n_files": len(results),
            "total_trajectories": sum(r.get("n_trajectories", 0) for r in results),
            "avg_trajectories": sum(r.get("n_trajectories", 0) for r in results) / max(len(results), 1),
            "files": [
                {
                    "file": r.get("file_name", "?"),
                    "snr_class": r.get("snr_class", "?"),
                    "n_trajectories": r.get("n_trajectories", 0),
                    "n_estimated": r.get("n_estimated", 0),
                    "peak_angles": r.get("peak_angles", []),
                    "validation": r.get("validation_report", {}).get("overall", "?"),
                    "processing_time": r.get("processing_time", 0)
                }
                for r in results
            ]
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
