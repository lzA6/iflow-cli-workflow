#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单清理重复文件脚本
安全地识别和清理明显重复的文件，保留最新版本
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKUP_DIR = PROJECT_ROOT / "cleanup_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")

def create_backup():
    """创建备份目录"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"创建备份目录: {BACKUP_DIR}")
    return True

def backup_file(file_path):
    """备份文件到备份目录"""
    try:
        source = PROJECT_ROOT / file_path
        if source.exists():
            backup_path = BACKUP_DIR / file_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, backup_path)
            print(f"已备份: {file_path}")
            return True
    except Exception as e:
        print(f"备份失败 {file_path}: {e}")
        return False

def analyze_duplicates():
    """分析重复文件"""
    print("分析重复文件...")
    
    # 需要清理的重复文件
    duplicate_files = [
        "iflow/cli_integration_v6.py",
        "iflow/hooks/enhanced_hooks_system_v7.py", 
        "iflow/hooks/enhanced_hooks_system_v8.py",
        "iflow/hooks/intelligent_hooks_system_v6.py",
        "iflow/hooks/intelligent_hooks_system_v8.py"
    ]
    
    files_to_remove = []
    for file_path in duplicate_files:
        source_file = PROJECT_ROOT / file_path
        if source_file.exists():
            files_to_remove.append(file_path)
    
    return files_to_remove

def show_analysis(files_to_remove):
    """显示分析结果"""
    print("\n重复文件分析结果:")
    print("=" * 60)
    
    if not files_to_remove:
        print("没有发现重复文件")
        return
    
    print(f"\n需要删除的文件 ({len(files_to_remove)}个):")
    for file_path in files_to_remove:
        source_file = PROJECT_ROOT / file_path
        if source_file.exists():
            size = source_file.stat().st_size
            print(f"  - {file_path} ({size} bytes)")
    
    total_size = sum((PROJECT_ROOT / f).stat().st_size for f in files_to_remove if (PROJECT_ROOT / f).exists())
    print(f"\n预估节省空间: {total_size} bytes")

def confirm_cleanup():
    """确认清理操作"""
    response = input("\n确认执行清理操作吗？(y/N): ").strip().lower()
    return response in ['y', 'yes', '是', '确认']

def execute_cleanup(files_to_remove):
    """执行清理操作"""
    print("\n开始执行清理...")
    
    removed_count = 0
    error_count = 0
    
    for file_path in files_to_remove:
        file_path_obj = PROJECT_ROOT / file_path
        
        try:
            # 先备份
            if not backup_file(file_path):
                error_count += 1
                continue
            
            # 删除文件
            file_path_obj.unlink()
            print(f"已删除: {file_path}")
            removed_count += 1
            
        except Exception as e:
            print(f"删除失败 {file_path}: {e}")
            error_count += 1
    
    return removed_count, error_count

def generate_cleanup_report(removed_count, error_count):
    """生成清理报告"""
    report = {
        "cleanup_date": datetime.now().isoformat(),
        "removed_files": removed_count,
        "failed_operations": error_count,
        "backup_location": str(BACKUP_DIR)
    }
    
    # 保存报告
    report_file = BACKUP_DIR / "cleanup_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n清理报告已保存: {report_file}")
    print(f"清理统计:")
    print(f"  - 删除文件: {removed_count}个")
    print(f"  - 失败操作: {error_count}个")
    print(f"  - 备份位置: {BACKUP_DIR}")

def main():
    """主函数"""
    print("简单清理重复文件脚本")
    print("=" * 60)
    
    # 创建备份
    if not create_backup():
        print("创建备份失败，退出")
        return False
    
    # 分析重复文件
    files_to_remove = analyze_duplicates()
    
    # 显示分析结果
    show_analysis(files_to_remove)
    
    # 如果没有找到重复文件
    if not files_to_remove:
        print("没有发现需要清理的重复文件")
        return True
    
    # 确认清理
    if not confirm_cleanup():
        print("用户取消清理操作")
        return True
    
    # 执行清理
    removed_count, error_count = execute_cleanup(files_to_remove)
    
    # 生成报告
    generate_cleanup_report(removed_count, error_count)
    
    print("\n清理完成！")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)