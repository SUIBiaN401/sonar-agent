"""
声纳信号处理自主Agent — 程序入口
"""

import os
import sys
import json
import time
import io
from pathlib import Path

# 修复 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 将项目根目录加入路径
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from interfaces.cli import parse_args
from agent.core_agent import SonarAgent


def main():
    args = parse_args()

    # 确定路径
    config_path = args.config or str(PROJECT_DIR / "config" / "default_config.json")
    workspace = args.workspace or str(PROJECT_DIR)
    output_dir = args.output or str(PROJECT_DIR / "output")

    print("=" * 60)
    print("  声纳信号处理自主Agent")
    print("=" * 60)
    print(f"  模式: {args.mode}")
    print(f"  配置: {config_path}")
    print(f"  工作区: {workspace}")
    print(f"  输出: {output_dir}")
    print("=" * 60)

    # 初始化Agent
    agent = SonarAgent(config_path, workspace)

    start_time = time.time()

    if args.mode == "inspect":
        # 数据检查模式
        data_dirs = args.data_dirs or [os.path.join("..", "data", "train")]
        for data_dir in data_dirs:
            print(f"\n[DIR] 检查数据目录: {data_dir}")
            report = agent.inspect(data_dir)
            print(f"  文件数: {report.get('n_files', 0)}")
            print(f"  SNR分布: {report.get('snr_distribution', {})}")
            for f in report.get("files", []):
                status = "✅" if f["status"] == "OK" else "❌"
                print(f"  {status} {f['file']}: SNR={f.get('snr_class', '?')}, "
                      f"std={f.get('db_std', '?')}, shape={f.get('shape', '?')}")

    elif args.mode == "single":
        # 单文件模式
        if not args.file:
            print("❌ 单文件模式需要 --file 参数")
            sys.exit(1)
        result = agent.process_single(args.file, output_dir)
        print(f"\n📊 处理结果:")
        print(f"  文件: {result['file_name']}")
        print(f"  SNR: {result['snr_class']} (std={result['db_std']})")
        print(f"  检测点: {result['n_detections']}")
        print(f"  聚类数: {result['n_clusters']}")
        print(f"  轨迹数: {result['n_trajectories']}")
        print(f"  估计目标数: {result['n_estimated']}")
        print(f"  峰值角度: {result['peak_angles']}")
        print(f"  验证: {result.get('validation_report', {}).get('overall', '?')}")
        print(f"  耗时: {result['processing_time']}s")

    elif args.mode == "batch":
        # 批量模式
        data_dirs = args.data_dirs or [
            os.path.join("..", "data", "train"),
            os.path.join("..", "data", "test")
        ]
        all_results = []
        for data_dir in data_dirs:
            print(f"\n[DIR] 处理目录: {data_dir}")
            results = agent.process_batch(data_dir, output_dir)
            all_results.extend(results)

        # 生成汇总报告
        md_path = os.path.join(output_dir, "report.md")
        json_path = os.path.join(output_dir, "report.json")
        agent.reporter.generate_markdown(all_results, md_path)
        agent.reporter.generate_json(all_results, json_path)

        print(f"\n📊 批量处理完成:")
        print(f"  总文件数: {len(all_results)}")
        print(f"  总轨迹数: {sum(r.get('n_trajectories', 0) for r in all_results)}")
        avg = sum(r.get('n_trajectories', 0) for r in all_results) / max(len(all_results), 1)
        print(f"  平均轨迹数/文件: {avg:.1f}")
        print(f"  报告: {md_path}")

    elif args.mode == "diagnose":
        # 诊断模式
        if not args.file:
            print("❌ 诊断模式需要 --file 参数")
            sys.exit(1)
        result = agent.process_single(args.file, output_dir)
        print(f"\n🔍 诊断结果:")
        print(f"  文件: {result['file_name']}")
        print(f"  估计目标数: {result['n_estimated']}")
        if args.n_targets:
            match = "✅" if result['n_estimated'] == args.n_targets else "❌"
            print(f"  {match} 与已知目标数({args.n_targets})比较: {result['n_estimated']}")
        print(f"  验证报告:")
        vr = result.get("validation_report", {})
        for check in vr.get("checks", []):
            icon = "✅" if check["status"] == "PASS" else "⚠️"
            print(f"    {icon} {check['name']}: {check['detail']}")

    elif args.mode == "validate":
        # 验证已有结果
        print("验证模式: 请指定结果JSON文件路径")

    total_time = time.time() - start_time
    print(f"\n⏱ 总耗时: {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
