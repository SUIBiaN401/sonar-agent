"""
数据检查工具：扫描数据文件，评估质量
"""

import os
from pathlib import Path
from typing import List, Dict

import numpy as np


class DataInspector:
    """数据检查器"""

    def __init__(self, config: dict):
        self.config = config

    def inspect(self, data_dir: str) -> dict:
        """
        扫描目录中的所有MAT文件
        """
        from pipeline.data_loader import DataLoader
        loader = DataLoader(self.config)

        files = loader.discover_files(data_dir)
        if not files:
            return {"status": "error", "message": f"未在 {data_dir} 中发现MAT文件"}

        file_infos = []
        for f in files:
            try:
                data = loader.load(f)
                file_infos.append({
                    "file": data["file_name"],
                    "shape": data["shape"],
                    "snr_class": data["snr_class"],
                    "db_std": data["db_std"],
                    "nan_ratio": data["nan_ratio"],
                    "status": "OK"
                })
            except Exception as e:
                file_infos.append({
                    "file": Path(f).stem,
                    "status": "ERROR",
                    "error": str(e)
                })

        # 统计
        snr_dist = {}
        for info in file_infos:
            cls = info.get("snr_class", "unknown")
            snr_dist[cls] = snr_dist.get(cls, 0) + 1

        return {
            "status": "ok",
            "n_files": len(file_infos),
            "snr_distribution": snr_dist,
            "files": file_infos
        }
