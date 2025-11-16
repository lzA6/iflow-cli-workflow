#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQ分析命令 V17 Hyperdimensional Singularity
=============================================

统一的 /arq-analysis 命令入口 - 超维奇点版

你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）
"""

import sys
import os
import subprocess
from pathlib import Path

# 获取脚本目录
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

# 添加到Python路径
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

def main():
    """主函数"""
    # 获取命令行参数
    args = sys.argv[1:]
    
    # 构建命令
    cmd = [sys.executable, str(script_dir / "arq-analysis-workflow-v17.py")] + args
    
    try:
        # 运行V17工作流
        result = subprocess.run(cmd, cwd=str(project_root))
        return result.returncode
    except Exception as e:
        print(f"错误: {e}")
        print("\n使用方法:")
        print("python .iflow/commands/arq-analysis \"查询内容\"")
        print("或")
        print("python .iflow/commands/arq-analysis-workflow-v17.py \"查询内容\"")
        return 1

if __name__ == "__main__":
    sys.exit(main())