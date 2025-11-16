#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/sc:test 指令的 MCP 服务器实现
提供增强版测试和分析功能的 MCP 工具接口
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from test_enhanced_main import EnhancedSCTestCommand
except ImportError:
    # 如果作为独立脚本运行
    from .test_enhanced_main import EnhancedSCTestCommand


class SCTestMCPServer:
    """SC Test MCP 服务器"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.test_command = EnhancedSCTestCommand(str(self.project_root))
        
    async def run_enhanced_test(self, 
                               interactive_mode: bool = True,
                               no_ai_awareness: bool = False,
                               no_deep_analysis: bool = False,
                               no_optimization_report: bool = False,
                               no_structure_comparison: bool = False) -> Dict[str, Any]:
        """运行增强版测试"""
        try:
            results = await self.test_command.execute_enhanced_test(
                interactive_mode=interactive_mode,
                force_ai_awareness=not no_ai_awareness,
                enable_deep_analysis=not no_deep_analysis,
                generate_optimization_report=not no_optimization_report,
                compare_structures=not no_structure_comparison
            )
            
            return {
                "success": True,
                "results": results,
                "message": "增强版测试执行完成",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "测试执行失败",
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_test_status(self) -> Dict[str, Any]:
        """获取测试状态"""
        try:
            # 检查项目结构
            reports_dir = self.project_root / "reports"
            reports_exist = reports_dir.exists()
            
            # 统计报告文件
            report_files = []
            if reports_exist:
                report_files = list(reports_dir.glob("*.json"))
            
            return {
                "success": True,
                "status": "ready",
                "reports_directory_exists": reports_exist,
                "report_files_count": len(report_files),
                "project_root": str(self.project_root),
                "available_features": [
                    "深度代码扫描",
                    "功能特点分析", 
                    "决策记录生成",
                    "优化报告生成",
                    "交互式分析"
                ]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "error"
            }
    
    async def show_help(self) -> Dict[str, Any]:
        """显示帮助信息"""
        help_content = """
# 🎯 增强版 /sc:test 系统帮助

## 🚀 快速开始
```bash
/sc:test                    # 启动交互模式
python .iflow/commands/sc/test_enhanced_main.py  # 直接运行
```

## 📋 核心功能
1. **深度代码扫描** - 全面的安全、性能、质量分析
2. **功能特点分析** - 智能价值评估和特点识别
3. **决策记录生成** - 基于证据的智能决策支持
4. **优化报告生成** - 自动化改进建议和实施计划
5. **交互式分析** - 友好的用户界面和深度探索

## 🛠️ 使用方法
- 交互模式：推荐用于深度分析和探索
- 非交互模式：适合自动化和批量处理
- 自定义配置：支持多种参数和选项

## 📊 输出报告
- JSON格式：结构化数据，便于程序处理
- Markdown格式：易读的报告，便于人工查看
- 位置：reports/ 目录

## 💡 提示
- 首次使用建议运行交互模式
- 可以随时中断和重新开始
- 所有分析结果都会保存到文件
        """
        
        return {
            "success": True,
            "help_content": help_content.strip(),
            "version": "2.0.0",
            "last_updated": datetime.now().isoformat()
        }


# MCP 服务器主函数
async def main():
    """MCP 服务器主函数"""
    server = SCTestMCPServer()
    
    # 简单的命令行界面
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "run":
            # 运行测试
            interactive = "--no-interactive" not in sys.argv
            result = await server.run_enhanced_test(interactive_mode=interactive)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
        elif command == "status":
            # 获取状态
            result = await server.get_test_status()
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
        elif command == "help":
            # 显示帮助
            result = await server.show_help()
            print(result["help_content"])
            
        else:
            print("可用命令: run, status, help")
    else:
        print("SC Test MCP Server - 使用 'help' 查看可用命令")


if __name__ == "__main__":
    asyncio.run(main())