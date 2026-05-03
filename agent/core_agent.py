"""
核心Agent类：大脑 — 任务规划、流程编排、自我验证、反思
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline import (
    DataLoader, CFARDetector, ClusterAnalyzer, KalmanTracker,
    QualityScorer, PeakVerifier, AngleJunctionSplitter,
    AggressiveMerger, WeakTargetRecovery, Visualizer
)
from validation import SelfValidator, CrossFileValidator, DiagnosticPlots
from tools import DataInspector, ReportGenerator
from agent.memory import AgentMemory
from agent.planner import TaskPlanner


class SonarAgent:
    """声纳信号处理自主Agent"""

    def __init__(self, config_path: str, workspace_dir: str):
        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)

        # 初始化记忆
        self.memory = AgentMemory(workspace_dir)

        # 初始化规划器
        self.planner = TaskPlanner()

        # 初始化Pipeline模块
        self.loader = DataLoader(self.config)
        self.cfar = CFARDetector(self.config)
        self.clusterer = ClusterAnalyzer(self.config)
        self.tracker = KalmanTracker(self.config)
        self.scorer = QualityScorer(self.config)
        self.verifier = PeakVerifier(self.config)
        self.splitter = AngleJunctionSplitter(self.config)
        self.merger = AggressiveMerger(self.config)
        self.recovery = WeakTargetRecovery(self.config)
        self.visualizer = Visualizer(self.config)

        # 初始化验证模块
        self.self_validator = SelfValidator(self.config)
        self.cross_validator = CrossFileValidator(self.config)
        self.diag_plots = DiagnosticPlots(self.config)

        # 初始化工具
        self.inspector = DataInspector(self.config)
        self.reporter = ReportGenerator(self.config)

        # 日志
        self._setup_logging()

        self.memory.append_journal(
            f"Agent初始化完成 | 工作区: {workspace_dir}",
            tags=["init"]
        )

    def _setup_logging(self):
        log_dir = self.workspace / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_dir / "agent.log", encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("SonarAgent")

    # ── 核心处理流程 ──────────────────────────────────────────────

    def process_single(self, file_path: str, output_dir: Optional[str] = None) -> dict:
        """
        单文件完整处理流程
        """
        start_time = time.time()
        self.logger.info(f"开始处理: {file_path}")

        if output_dir is None:
            output_dir = str(self.workspace / "output")

        try:
            # Stage 1: 加载数据
            data = self.loader.load(file_path)
            self.logger.info(f"  加载完成: {data['file_name']} | SNR={data['snr_class']} | shape={data['shape']}")

            # Stage 2: CFAR检测
            cfar_result = self.cfar.detect(data["data"], data["snr_class"])
            n_dets = len(cfar_result["detections"])
            self.logger.info(f"  CFAR检测: {n_dets}个检测点")

            # Stage 3: 角度跳跃分割
            dets = cfar_result["detections"]
            if len(dets) > 0:
                segments = self.splitter.split(dets)
                # 合并所有分割后的检测点
                if segments:
                    dets = np.vstack(segments) if len(segments) > 1 else segments[0]
                else:
                    dets = np.array([]).reshape(0, 2)
            self.logger.info(f"  角度分割后: {len(dets)}个检测点")

            # Stage 4: DBSCAN聚类
            cluster_result = self.clusterer.cluster(dets)
            self.logger.info(f"  聚类结果: {cluster_result['n_clusters']}个聚类")

            # Stage 5: Kalman跟踪
            trajectories = self.tracker.track(cluster_result, cfar_result["residual"])
            self.logger.info(f"  跟踪结果: {len(trajectories)}条轨迹")

            # Stage 6: 激进合并
            trajectories = self.merger.merge(trajectories)
            self.logger.info(f"  合并后: {len(trajectories)}条轨迹")

            # Stage 7: 弱目标恢复
            recovered = self.recovery.recover(
                data["data"], trajectories, cfar_result, cfar_result["residual"]
            )
            if recovered:
                trajectories.extend(recovered)
                self.logger.info(f"  弱目标恢复: +{len(recovered)}条")

            # Stage 8: 质量评分
            quality_scores = []
            for traj in trajectories:
                qs = self.scorer.score(traj, cfar_result["residual"])
                traj["quality_score"] = qs
                quality_scores.append(qs)
            self.logger.info(f"  质量评分: {[q['total'] for q in quality_scores]}")

            # Stage 9: 峰值验证
            peak_result = self.verifier.verify(
                trajectories, cfar_result["residual"], cfar_result["mask"]
            )
            valid_trajectories = peak_result["valid_trajectories"]
            self.logger.info(f"  峰值验证: 估计{peak_result['n_estimated']}个目标, {len(valid_trajectories)}条有效轨迹")

            # 构建结果
            result = {
                "file_name": data["file_name"],
                "file_path": file_path,
                "snr_class": data["snr_class"],
                "db_std": data["db_std"],
                "n_detections": n_dets,
                "n_clusters": cluster_result["n_clusters"],
                "n_trajectories": len(valid_trajectories),
                "n_estimated": peak_result["n_estimated"],
                "peak_angles": peak_result["peak_angles"],
                "trajectories": valid_trajectories,
                "quality_scores": quality_scores,
                "peak_result": peak_result,
                "residual": cfar_result["residual"],
                "processing_time": round(time.time() - start_time, 2)
            }

            # Stage 10: 可视化
            vis_path = os.path.join(output_dir, f"{data['file_name']}_result.png")
            self.visualizer.plot_full(
                data, cfar_result, cluster_result,
                valid_trajectories, quality_scores,
                peak_result, vis_path
            )
            result["visualization_path"] = vis_path
            self.logger.info(f"  可视化已保存: {vis_path}")

            # Stage 11: 自我验证
            validation = self.self_validator.validate(result)
            result["validation_report"] = validation
            self.logger.info(f"  自我验证: {validation['summary']}")

            # 验证失败时生成诊断图
            if validation["overall"] != "PASS":
                diag_dir = os.path.join(output_dir, "diagnostics")
                diag_files = self.diag_plots.generate(
                    data, cfar_result, peak_result, diag_dir
                )
                result["diagnostic_files"] = diag_files
                self.logger.info(f"  诊断图已生成: {len(diag_files)}张")

            # 记录到日志
            self.memory.append_journal(
                f"处理 {data['file_name']}: {len(valid_trajectories)}条轨迹, "
                f"估计{peak_result['n_estimated']}个目标, "
                f"验证={validation['overall']}",
                tags=["process", data["snr_class"]]
            )

            # 保存结果JSON
            json_path = os.path.join(output_dir, f"{data['file_name']}_result.json")
            self._save_result(result, json_path)

            self.logger.info(f"处理完成: {data['file_name']} ({result['processing_time']}s)")
            return result

        except Exception as e:
            self.logger.error(f"处理失败: {file_path} | {str(e)}")
            self.memory.append_journal(
                f"处理失败 {file_path}: {str(e)}",
                tags=["error"]
            )
            raise

    def process_batch(self, data_dir: str, output_dir: Optional[str] = None) -> list[dict]:
        """
        批量处理
        """
        files = self.loader.discover_files(data_dir)
        self.logger.info(f"批量处理: 发现{len(files)}个文件")

        if output_dir is None:
            output_dir = str(self.workspace / "output")

        results = []
        for f in files:
            try:
                result = self.process_single(f, output_dir)
                results.append(result)
            except Exception as e:
                self.logger.error(f"跳过文件 {f}: {e}")
                results.append({
                    "file_path": f,
                    "error": str(e),
                    "n_trajectories": 0
                })

        # 跨文件验证
        cross_report = self.cross_validator.validate(results)
        self.logger.info(f"跨文件验证: {cross_report['summary']}")

        # 生成汇总报告
        report_path = os.path.join(output_dir, "batch_report.json")
        with open(report_path, "w", encoding="utf-8") as fp:
            json.dump({
                "results": [
                    {
                        "file": r.get("file_name", "?"),
                        "n_traj": r.get("n_trajectories", 0),
                        "n_est": r.get("n_estimated", 0),
                        "snr": r.get("snr_class", "?"),
                        "validation": r.get("validation_report", {}).get("overall", "?"),
                        "time": r.get("processing_time", 0)
                    }
                    for r in results
                ],
                "cross_validation": cross_report
            }, fp, ensure_ascii=False, indent=2)

        self.memory.append_journal(
            f"批量处理完成: {len(results)}个文件, "
            f"平均{sum(r.get('n_trajectories',0) for r in results)/max(len(results),1):.1f}条轨迹/文件",
            tags=["batch"]
        )

        return results

    def inspect(self, data_dir: str) -> dict:
        """数据检查"""
        return self.inspector.inspect(data_dir)

    def _save_result(self, result: dict, path: str):
        """保存结果（序列化友好格式）"""
        serializable = {
            k: v for k, v in result.items()
            if k not in ("trajectories", "residual")
        }
        serializable["trajectories"] = [
            {
                "n_points": len(t.get("trajectory", [])),
                "t_start": t.get("t_start", 0),
                "t_end": t.get("t_end", 0),
                "valid_count": t.get("valid_count", 0),
                "quality": t.get("quality_score", {}),
                "mean_angle": float(np.mean(t["trajectory"][:, 1])) if len(t.get("trajectory", [])) > 0 else None
            }
            for t in result.get("trajectories", [])
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)


