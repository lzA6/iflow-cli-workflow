#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接执行的全自动化项目审查和升级命令
Direct Comprehensive Project Upgrade Command
"""

import os
import sys
import asyncio
import argparse
import json
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

async def execute_comprehensive_upgrade(workspace=".", auto_fix=True, no_backup=False, dry_run=False, verbose=False):
    """直接执行全面升级"""
    print("🚀 开始执行全自动化项目审查和升级...")
    
    try:
        # 导入工作流
        sys.path.insert(0, str(current_dir))
        import importlib.util
        spec = importlib.util.spec_from_file_location("comprehensive_project_upgrade_workflow", current_dir / "comprehensive-project-upgrade-workflow.py")
        workflow_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow_module)
        ComprehensiveProjectUpgradeWorkflow = workflow_module.ComprehensiveProjectUpgradeWorkflow
        
        # 配置
        config = {
            "auto_fix": auto_fix,
            "backup_enabled": not no_backup,
            "analysis_mode": dry_run,
            "verbose": verbose
        }
        
        # 创建工作流
        workflow = ComprehensiveProjectUpgradeWorkflow(workspace, config)
        
        print("📊 阶段1: 深度分析...")
        
        # 初始化
        await workflow.initialize()
        
        print("🔧 阶段2-6: 执行升级流程...")
        
        # 执行升级
        report = await workflow.execute_comprehensive_upgrade()
        
        # 输出结果
        summary = report["analysis_summary"]
        print(f"\n🎉 项目升级完成!")
        print(f"📊 总问题数: {summary['total_issues_detected']}")
        print(f"✅ 检测完成: {summary['total_issues_detected']} 个问题")
        print(f"🔍 分析模式: 检测报告生成")
        print(f"⏱️ 耗时: {summary['duration_minutes']:.1f} 分钟")
        print(f"📋 详细报告: .iflow/reports/upgrade_report_{workflow.session_id}.md")
        
        return 0
        
    except Exception as e:
        print(f"❌ 升级失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="全自动化项目审查和升级")
    parser.add_argument("--workspace", "-w", default=".", help="工作空间路径")
    parser.add_argument("--auto-fix", action="store_true", default=True, help="自动修复问题")
    parser.add_argument("--no-backup", action="store_true", help="不创建备份")
    parser.add_argument("--dry-run", action="store_true", help="分析模式，仅生成报告不修改文件")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 执行升级
    exit_code = asyncio.run(execute_comprehensive_upgrade(
        workspace=args.workspace,
        auto_fix=args.auto_fix,
        no_backup=args.no_backup,
        dry_run=args.dry_run,
        verbose=args.verbose
    ))
    
    sys.exit(exit_code)
else:
    # 当被导入时直接执行
    exit_code = asyncio.run(execute_comprehensive_upgrade())
    sys.exit(exit_code)
