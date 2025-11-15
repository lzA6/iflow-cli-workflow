#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能重复文件清理器 - 安全删除重复和旧版本文件
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import hashlib

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class IntelligentDuplicateRemover:
    """智能重复文件清理器"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.backup_dir = self.project_root / ".backup" / f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.deleted_files = []
        self.skipped_files = []
        
    def get_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                buf = f.read(8192)
                while buf:
                    hasher.update(buf)
                    buf = f.read(8192)
            return hasher.hexdigest()
        except:
            return ""
    
    def find_duplicate_files(self) -> Dict[str, List[Path]]:
        """查找重复文件"""
        print("🔍 正在扫描重复文件...")
        
        file_hashes = {}
        duplicates = {}
        
        # 扫描scripts目录
        scripts_dir = self.project_root / "scripts"
        if scripts_dir.exists():
            for file_path in scripts_dir.glob("*.py"):
                if file_path.is_file():
                    file_hash = self.get_file_hash(file_path)
                    if file_hash:
                        if file_hash in file_hashes:
                            if file_hash not in duplicates:
                                duplicates[file_hash] = [file_hashes[file_hash]]
                            duplicates[file_hash].append(file_path)
                        else:
                            file_hashes[file_hash] = file_path
        
        return duplicates
    
    def identify_removable_files(self) -> List[Path]:
        """识别可安全删除的文件"""
        print("🤖 正在识别可删除文件...")
        
        removable_files = []
        
        # 1. 旧版本批处理脚本
        old_batch_files = [
            "install_tools_v2.bat",
            "start_tools_v2.bat", 
            "install_tools.bat",
            "start_tools.bat"
        ]
        
        for file_name in old_batch_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                removable_files.append(file_path)
                print(f"  发现旧版批处理: {file_name}")
        
        # 2. 重复的智能版本分析器
        duplicate_pairs = [
            ("smart_version_analyzer.py", "smart_version_analyzer_fixed.py"),
        ]
        
        for keep_file, remove_file in duplicate_pairs:
            keep_path = self.project_root / "scripts" / keep_file
            remove_path = self.project_root / "scripts" / remove_file
            if keep_path.exists() and remove_path.exists():
                # 比较文件内容
                if self.get_file_hash(keep_path) == self.get_file_hash(remove_path):
                    removable_files.append(remove_path)
                    print(f"  发现重复文件: {remove_file} (与 {keep_file} 相同)")
                else:
                    # 保留较新的文件
                    keep_mtime = keep_path.stat().st_mtime
                    remove_mtime = remove_path.stat().st_mtime
                    if remove_mtime < keep_mtime:
                        removable_files.append(remove_path)
                        print(f"  发现旧版本: {remove_file}")
        
        # 3. 旧版本清理脚本
        old_cleanup_scripts = [
            "cleanup_v4.py",
            "cleanup_old_files.py", 
            "simple_cleanup.py",
            "conservative_cleanup_v2.py",  # 如果有更新的版本
            "fixed_cleanup_script.py"
        ]
        
        scripts_dir = self.project_root / "scripts"
        for file_name in old_cleanup_scripts:
            file_path = scripts_dir / file_name
            if file_path.exists():
                # 检查是否有更新的版本
                base_name = file_name.replace("_v4", "").replace("_v2", "").replace("_old", "")
                newer_versions = list(scripts_dir.glob(f"{base_name.replace('.py', '')}*.py"))
                if len(newer_versions) > 1:
                    # 按修改时间排序，保留最新的
                    newer_versions.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    for old_file in newer_versions[1:]:
                        if old_file not in removable_files:
                            removable_files.append(old_file)
                            print(f"  发现旧版本清理脚本: {old_file.name}")
        
        # 4. 核心目录中的旧版本文件
        core_dir = self.project_root / "core"
        if core_dir.exists():
            # 查找版本号较高的文件
            versioned_files = {}
            for file_path in core_dir.glob("*.py"):
                if "_v" in file_path.name:
                    base_name = file_path.name.split("_v")[0]
                    if base_name not in versioned_files:
                        versioned_files[base_name] = []
                    versioned_files[base_name].append(file_path)
            
            # 对每个基础名称，只保留最新版本
            for base_name, files in versioned_files.items():
                if len(files) > 1:
                    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    for old_file in files[1:]:
                        removable_files.append(old_file)
                        print(f"  发现旧版本核心文件: {old_file.name}")
        
        return removable_files
    
    def backup_file(self, file_path: Path):
        """备份文件"""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = self.backup_dir / file_path.name
            shutil.copy2(file_path, backup_path)
            print(f"  已备份: {file_path.name}")
            return True
        except Exception as e:
            print(f"  备份失败 {file_path.name}: {e}")
            return False
    
    def delete_files(self, files_to_delete: List[Path]):
        """删除文件"""
        print(f"\n🗑️  准备删除 {len(files_to_delete)} 个文件...")
        
        for file_path in files_to_delete:
            try:
                if file_path.exists():
                    # 先备份
                    if self.backup_file(file_path):
                        # 删除文件
                        if file_path.is_file():
                            file_path.unlink()
                        else:
                            shutil.rmtree(file_path)
                        
                        self.deleted_files.append(file_path)
                        print(f"  ✓ 已删除: {file_path.relative_to(self.project_root)}")
                    else:
                        self.skipped_files.append((file_path, "备份失败"))
                else:
                    self.skipped_files.append((file_path, "文件不存在"))
            except Exception as e:
                self.skipped_files.append((file_path, f"删除失败: {e}"))
                print(f"  ✗ 删除失败 {file_path.name}: {e}")
    
    def generate_report(self):
        """生成清理报告"""
        report = {
            "清理时间": datetime.now().isoformat(),
            "项目根目录": str(self.project_root),
            "备份目录": str(self.backup_dir),
            "删除文件数": len(self.deleted_files),
            "跳过文件数": len(self.skipped_files),
            "删除的文件": [str(f.relative_to(self.project_root)) for f in self.deleted_files],
            "跳过的文件": [(str(f.relative_to(self.project_root)), reason) for f, reason in self.skipped_files]
        }
        
        # 保存报告
        report_file = self.backup_dir / "cleanup_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 清理报告已保存: {report_file}")
        
        # 显示摘要
        print(f"\n📊 清理摘要:")
        print(f"  删除文件: {len(self.deleted_files)} 个")
        print(f"  跳过文件: {len(self.skipped_files)} 个")
        print(f"  备份位置: {self.backup_dir}")
        
        if self.deleted_files:
            print(f"\n🗑️  已删除的文件:")
            for file_path in self.deleted_files:
                print(f"    - {file_path.relative_to(self.project_root)}")
        
        if self.skipped_files:
            print(f"\n⚠️  跳过的文件:")
            for file_path, reason in self.skipped_files:
                print(f"    - {file_path.relative_to(self.project_root)}: {reason}")
    
    def run(self):
        """运行清理流程"""
        print("=" * 70)
        print("🧹 智能重复文件清理器")
        print("=" * 70)
        
        # 1. 查找重复文件
        duplicates = self.find_duplicate_files()
        if duplicates:
            print(f"\n📋 发现 {len(duplicates)} 组重复文件")
        
        # 2. 识别可删除文件
        removable_files = self.identify_removable_files()
        
        if not removable_files:
            print("\n✨ 未发现需要删除的文件")
            return
        
        print(f"\n🤖 识别出 {len(removable_files)} 个可删除文件:")
        for file_path in removable_files:
            print(f"  - {file_path.relative_to(self.project_root)}")
        
        # 3. 确认删除
        print(f"\n⚠️  即将删除 {len(removable_files)} 个文件")
        print(f"   备份位置: {self.backup_dir}")
        
        try:
            confirm = input("\n确认删除? (yes/no): ").strip().lower()
            if confirm == 'yes':
                # 4. 删除文件
                self.delete_files(removable_files)
                
                # 5. 生成报告
                self.generate_report()
                
                print("\n✅ 清理完成！")
            else:
                print("\n❌ 操作已取消")
        except KeyboardInterrupt:
            print("\n\n❌ 用户中断操作")
        except Exception as e:
            print(f"\n❌ 操作失败: {e}")


def main():
    """主函数"""
    cleaner = IntelligentDuplicateRemover()
    cleaner.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 用户中断操作")
    except Exception as e:
        print(f"\n❌ 系统运行失败: {e}")
        import traceback
        traceback.print_exc()
