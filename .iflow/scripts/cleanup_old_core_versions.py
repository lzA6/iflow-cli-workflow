#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理旧版本核心文件 - 分析并删除旧版本
"""

import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


def analyze_versioned_files():
    """分析版本化文件"""
    print("🔍 分析core目录中的版本化文件...")
    
    core_dir = PROJECT_ROOT / "core"
    if not core_dir.exists():
        print("❌ core目录不存在")
        return
    
    files = list(core_dir.glob("*.py"))
    
    # 按基础名称分组
    file_groups = {}
    for file in files:
        # 提取基础名称（去掉版本号）
        base_name = re.sub(r'_v\d+', '', file.name)
        base_name = re.sub(r'_v\d+_\w+', '', base_name)
        
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append(file)
    
    # 查找有多个版本的文件
    versioned_files = {}
    for base_name, file_list in file_groups.items():
        if len(file_list) > 1:
            versioned_files[base_name] = file_list
    
    if not versioned_files:
        print("✨ 未发现多版本文件")
        return
    
    print(f"\n📊 发现 {len(versioned_files)} 组多版本文件:")
    
    removable_files = []
    
    for base_name, file_list in versioned_files.items():
        print(f"\n{base_name}: {len(file_list)} 个版本")
        
        # 按修改时间排序（最新的在前）
        sorted_files = sorted(file_list, key=lambda x: x.stat().st_mtime, reverse=True)
        
        for i, f in enumerate(sorted_files):
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            size_kb = f.stat().st_size / 1024
            marker = " → 保留" if i == 0 else " → 可删除"
            print(f"  {i+1}. {f.name:<50} {mtime.strftime('%Y-%m-%d %H:%M')} {size_kb:>8.1f}KB{marker}")
            
            # 保留最新版本，删除旧版本
            if i > 0:
                removable_files.append(f)
    
    return removable_files


def create_backup(removable_files: List[Path]):
    """创建备份"""
    backup_dir = PROJECT_ROOT / ".backup" / f"core_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 正在备份 {len(removable_files)} 个文件到 {backup_dir}...")
    
    for file_path in removable_files:
        try:
            shutil.copy2(file_path, backup_dir / file_path.name)
            print(f"  ✓ 备份: {file_path.name}")
        except Exception as e:
            print(f"  ✗ 备份失败 {file_path.name}: {e}")
    
    return backup_dir


def delete_files(removable_files: List[Path], backup_dir: Path):
    """删除文件"""
    print(f"\n🗑️  正在删除 {len(removable_files)} 个旧版本文件...")
    
    deleted_files = []
    failed_files = []
    
    for file_path in removable_files:
        try:
            file_path.unlink()
            deleted_files.append(file_path.name)
            print(f"  ✓ 删除: {file_path.name}")
        except Exception as e:
            failed_files.append((file_path.name, str(e)))
            print(f"  ✗ 删除失败 {file_path.name}: {e}")
    
    return deleted_files, failed_files


def generate_report(backup_dir: Path, deleted_files: List[str], failed_files: List[Tuple[str, str]]):
    """生成清理报告"""
    print(f"\n📄 生成清理报告...")
    
    report = {
        "清理时间": datetime.now().isoformat(),
        "备份目录": str(backup_dir),
        "删除文件数": len(deleted_files),
        "失败文件数": len(failed_files),
        "删除的文件": deleted_files,
        "失败的文件": failed_files
    }
    
    # 计算释放的空间
    total_size_mb = sum((backup_dir / f).stat().st_size for f in deleted_files) / 1024 / 1024
    
    print(f"\n" + "=" * 70)
    print("📊 清理报告")
    print("=" * 70)
    print(f"清理时间: {report['清理时间']}")
    print(f"备份目录: {backup_dir}")
    print(f"删除文件: {len(deleted_files)} 个")
    print(f"失败文件: {len(failed_files)} 个")
    print(f"释放空间: {total_size_mb:.2f} MB")
    
    if deleted_files:
        print(f"\n🗑️  已删除的文件:")
        for filename in deleted_files:
            print(f"  - {filename}")
    
    if failed_files:
        print(f"\n❌ 删除失败的文件:")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    
    # 保存报告
    report_file = backup_dir / "core_cleanup_report.json"
    import json
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 报告已保存: {report_file}")
    print("=" * 70)


def main():
    """主函数"""
    print("=" * 70)
    print("🧹 旧版本核心文件清理器")
    print("=" * 70)
    
    # 1. 分析版本化文件
    removable_files = analyze_versioned_files()
    
    if not removable_files:
        print("\n✨ 无需清理")
        return
    
    print(f"\n⚠️  发现 {len(removable_files)} 个可删除的旧版本文件")
    
    try:
        # 2. 确认删除
        confirm = input("\n确认删除这些旧版本文件? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("\n❌ 操作已取消")
            return
        
        # 3. 创建备份
        backup_dir = create_backup(removable_files)
        
        # 4. 删除文件
        deleted_files, failed_files = delete_files(removable_files, backup_dir)
        
        # 5. 生成报告
        generate_report(backup_dir, deleted_files, failed_files)
        
        print("\n✅ 清理完成！")
        
    except KeyboardInterrupt:
        print("\n\n🛑 用户中断操作")
    except Exception as e:
        print(f"\n❌ 操作失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
