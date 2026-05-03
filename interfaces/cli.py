"""
命令行接口
"""

import argparse
import sys


def parse_args(args=None):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="声纳信号处理自主Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --mode inspect --data-dir ../骗小米/data/训练
  python main.py --mode single --file ../骗小米/data/训练/q6-CBFresult.mat
  python main.py --mode batch --data-dir ../骗小米/data/训练
  python main.py --mode batch --data-dir ../骗小米/data/训练 --data-dir2 ../骗小米/data/测试
        """
    )

    parser.add_argument("--mode", required=True,
                        choices=["inspect", "single", "batch", "diagnose", "validate"],
                        help="运行模式")
    parser.add_argument("--file", type=str, help="单文件模式下的MAT文件路径")
    parser.add_argument("--data-dir", type=str, action="append", dest="data_dirs",
                        help="数据目录（可指定多个）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录（默认: workspace/output）")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径（默认: config/default_config.json）")
    parser.add_argument("--workspace", type=str, default=None,
                        help="工作区目录（默认: 当前目录）")
    parser.add_argument("--n-targets", type=int, default=None,
                        help="诊断模式下的已知目标数")

    return parser.parse_args(args)
