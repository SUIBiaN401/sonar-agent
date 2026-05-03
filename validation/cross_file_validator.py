"""
跨文件验证器：批量结果一致性检查
"""

import json
from pathlib import Path
from typing import List, Dict


class CrossFileValidator:
    """跨文件结果验证器"""

    def __init__(self, config: dict):
        self.config = config

    def validate(self, results: List[dict]) -> dict:
        """
        验证批量处理结果的一致性
        """
        if not results:
            return {"overall": "FAIL", "summary": "无结果可验证", "checks": []}

        checks = []

        # 1. 目标数分布
        n_targets = [r.get("n_trajectories", 0) for r in results]
        avg_targets = sum(n_targets) / len(n_targets) if n_targets else 0
        checks.append({
            "name": "平均目标数",
            "status": "PASS" if 1.0 <= avg_targets <= 3.0 else "WARN",
            "detail": f"平均={avg_targets:.1f}/文件"
        })

        # 2. 零目标文件检查
        zero_files = [r.get("file_name", "?") for r in results if r.get("n_trajectories", 0) == 0]
        checks.append({
            "name": "零目标文件",
            "status": "PASS" if len(zero_files) <= 2 else "WARN",
            "detail": f"{len(zero_files)}个文件无目标: {zero_files}"
        })

        # 3. 异常多目标文件
        multi_files = [r.get("file_name", "?") for r in results if r.get("n_trajectories", 0) > 5]
        checks.append({
            "name": "过多目标文件",
            "status": "PASS" if len(multi_files) == 0 else "WARN",
            "detail": f"{len(multi_files)}个文件目标>5: {multi_files}"
        })

        # 4. 验证通过率
        pass_rates = []
        for r in results:
            vr = r.get("validation_report", {})
            if vr:
                pass_rates.append(vr.get("pass_rate", 0))
        avg_pass = sum(pass_rates) / len(pass_rates) if pass_rates else 0
        checks.append({
            "name": "平均验证通过率",
            "status": "PASS" if avg_pass >= 0.7 else "WARN",
            "detail": f"{avg_pass:.1%}"
        })

        # 汇总
        n_pass = sum(1 for c in checks if c["status"] == "PASS")
        n_warn = sum(1 for c in checks if c["status"] == "WARN")
        total = len(checks)
        overall = "FAIL" if n_warn > total * 0.5 else ("WARN" if n_warn > 0 else "PASS")

        return {
            "overall": overall,
            "checks": checks,
            "summary": f"跨文件验证: {n_pass}/{total} 通过",
            "n_files": len(results),
            "avg_targets": round(avg_targets, 2)
        }
