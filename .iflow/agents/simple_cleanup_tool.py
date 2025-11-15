#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版智能体清理工具
避免Unicode编码问题
"""

import os
import shutil
from pathlib import Path

def find_duplicate_files():
    """查找重复的.md文件"""
    agents_dir = Path(".")
    duplicates_found = []
    
    print("Finding duplicate .md files...")
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(agents_dir):
        root_path = Path(root)
        
        # 跳过根目录和隐藏目录
        if root_path == agents_dir or any(part.startswith('.') for part in root_path.parts):
            continue
            
        # 查找.md文件
        md_files = list(root_path.glob("*.md"))
        
        if len(md_files) > 1:
            # 如果有README.md，其他的都是重复
            readme_file = root_path / "README.md"
            if readme_file.exists():
                duplicates = [f for f in md_files if f.name != "README.md"]
                if duplicates:
                    duplicates_found.append((root_path, duplicates))
                    print(f"Found duplicates in {root_path}: {[f.name for f in duplicates]}")
    
    return duplicates_found

def merge_duplicate_files(agent_dir, duplicate_files):
    """合并重复文件到README.md"""
    readme_file = agent_dir / "README.md"
    
    print(f"Processing {agent_dir.name}...")
    
    # 读取现有README.md内容
    existing_content = ""
    if readme_file.exists():
        existing_content = readme_file.read_text(encoding='utf-8')
    
    # 合并所有重复文件的内容
    merged_content = [existing_content]
    
    for dup_file in duplicate_files:
        print(f"  Merging {dup_file.name}...")
        try:
            content = dup_file.read_text(encoding='utf-8')
            merged_content.append(f"\n\n## Content from {dup_file.name}\n\n{content}")
            
            # 备份并删除重复文件
            backup_name = f"{dup_file.stem}_backup{dup_file.suffix}"
            backup_path = dup_file.with_name(backup_name)
            shutil.copy2(dup_file, backup_path)
            dup_file.unlink()
            print(f"    Backup created: {backup_name}")
            
        except Exception as e:
            print(f"    Error processing {dup_file.name}: {e}")
    
    # 写回合并后的内容
    final_content = "\n".join(merged_content)
    readme_file.write_text(final_content, encoding='utf-8')
    print(f"  Updated README.md")

def standardize_readme(agent_dir):
    """标准化README.md格式"""
    readme_file = agent_dir / "README.md"
    
    if not readme_file.exists():
        print(f"  Creating new README.md for {agent_dir.name}")
        # 创建基本的README.md
        basic_content = f"""# {agent_dir.name.replace('-', ' ').title()}

## Overview
{agent_dir.name} agent provides specialized capabilities.

## Usage
Basic command: /{agent_dir.name.replace('-', '_')}

## Features
- Professional analysis
- Intelligent processing  
- Automated solutions

Created: 2025-11-15
"""
        readme_file.write_text(basic_content, encoding='utf-8')
        return
    
    # 如果README.md存在，检查是否需要添加标准内容
    content = readme_file.read_text(encoding='utf-8')
    
    # 添加基本结构如果不存在
    if "## Overview" not in content and "## 概述" not in content:
        content += "\n\n## Overview\nThis agent provides intelligent analysis and processing capabilities."
        readme_file.write_text(content, encoding='utf-8')
        print(f"  Added standard sections to README.md")

def main():
    """主函数"""
    print("Agent Cleanup Tool - Simplified Version")
    print("=" * 50)
    
    # 1. 查找重复文件
    duplicates = find_duplicate_files()
    
    if not duplicates:
        print("No duplicate files found.")
        return
    
    print(f"\nFound {len(duplicates)} directories with duplicate files")
    
    # 2. 处理每个目录
    total_merged = 0
    total_deleted = 0
    
    for agent_dir, duplicate_files in duplicates:
        print(f"\nProcessing: {agent_dir}")
        
        # 合并重复文件
        merge_duplicate_files(agent_dir, duplicate_files)
        total_merged += len(duplicate_files)
        total_deleted += len(duplicate_files)
        
        # 标准化README.md
        standardize_readme(agent_dir)
    
    # 3. 生成报告
    print("\n" + "=" * 50)
    print("CLEANUP COMPLETED")
    print("=" * 50)
    print(f"Directories processed: {len(duplicates)}")
    print(f"Files merged: {total_merged}")
    print(f"Files deleted: {total_deleted}")
    print(f"Backups created: {total_deleted}")
    print("\nNext steps:")
    print("1. Review the merged README.md files")
    print("2. Customize content for each agent")
    print("3. Test /command functionality")
    print("4. Remove backup files after validation")

if __name__ == "__main__":
    main()