#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库服务启动脚本
"""

import os
import sys
import json
import time
from pathlib import Path

# 内存优化设置
os.environ['PYTHONOPTIMIZE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    """主函数"""
    # 导入并启动Flask应用
    sys.path.insert(0, str(project_root / "knowledge_base" / "web_ui"))
    
    # 优化Flask配置
    os.environ['FLASK_ENV'] = 'production'
    os.environ['WERKZEUG_RUN_MAIN'] = 'true'
    
    from app import app
    
    # 启动配置
    config = {
        "host": "127.0.0.1",
        "port": 5000,
        "debug": False,
        "threaded": True,
        "use_reloader": False
    }
    
    try:
        app.run(**config)
    except KeyboardInterrupt:
        print("\n知识库服务已停止")
    except Exception as e:
        print(f"知识库服务错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
