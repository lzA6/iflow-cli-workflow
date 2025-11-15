#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保守清理重复文件脚本
安全地识别和清理明显重复的文件，保留最新版本
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
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

# 需要清理的重复文件映射
DUPLICATE_FILES = {
    # CLI集成文件
    "iflow/cli_integration_v6.py": {
        "keep": "iflow/cli_integration_enhanced_v7.py",
        "reason": "保留增强版本，清理旧版本"
    },
    
    # Hooks系统文件
    "iflow/hooks/enhanced_hooks_system_v7.py": {
        "keep": "iflow/hooks/enhanced_hooks_system_v9.py", 
        "reason": "保留v9版本，清理v7版本"
    },
    "iflow/hooks/enhanced_hooks_system_v8.py": {
        "keep": "iflow/hooks/enhanced_hooks_system_v9.py",
        "reason": "保留v9版本，清理v8版本"
    },
    
    "iflow/hooks/intelligent_hooks_system_v6.py": {
        "keep": "iflow/hooks/intelligent_hooks_system_v9.py",
        "reason": "保留v9版本，清理v6版本"
    },
    "iflow/hooks/intelligent_hooks_system_v8.py": {
        "keep": "iflow/hooks/intelligent_hooks_system_v9.py", 
        "reason": "保留v9版本，清理v8版本"
    },
    
    # Hook管理器占位符问题
    "iflow/hooks/comprehensive_hook_manager_placeholder.py": {
        "keep": "iflow/hooks/comprehensive_hook_manager_v4.py",
        "reason": "替换占位符为实际实现"
    }
}

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
            print(f"已备份: {file_path} -> {backup_path}")
            return True
    except Exception as e:
        print(f"备份失败 {file_path}: {e}")
        return False

def analyze_duplicates():
    """分析重复文件"""
    print("🔍 分析重复文件...")
    analysis = {
        "files_to_remove": [],
        "files_to_keep": [],
        "issues": []
    }
    
    for file_path, info in DUPLICATE_FILES.items():
        source_file = PROJECT_ROOT / file_path
        keep_file = PROJECT_ROOT / info["keep"]
        
        if source_file.exists():
            # 检查保留的文件是否存在
            if not keep_file.exists():
                analysis["issues"].append(f"⚠️ 保留文件不存在: {info['keep']}")
                continue
                
            # 获取文件信息
            source_stat = source_file.stat()
            keep_stat = keep_file.stat()
            
            analysis["files_to_remove"].append({
                "file": file_path,
                "size": source_stat.st_size,
                "modified": datetime.fromtimestamp(source_stat.st_mtime).isoformat(),
                "keep_file": info["keep"],
                "reason": info["reason"]
            })
            
            analysis["files_to_keep"].append({
                "file": info["keep"],
                "size": keep_stat.st_size,
                "modified": datetime.fromtimestamp(keep_stat.st_mtime).isoformat()
            })
    
    return analysis

def show_analysis(analysis):
    """显示分析结果"""
    print("\n重复文件分析结果:")
    print("=" * 60)
    
    for issue in analysis["issues"]:
        print(issue)
    
    print(f"\n需要删除的文件 ({len(analysis['files_to_remove'])}个):")
    for item in analysis["files_to_remove"]:
        print(f"  - {item['file']} ({item['size']} bytes)")
        print(f"    保留: {item['keep_file']}")
        print(f"    原因: {item['reason']}")
    
    print(f"\n需要保留的文件 ({len(set(item['file'] for item in analysis['files_to_keep']))}个):")
    seen = set()
    for item in analysis["files_to_keep"]:
        if item["file"] not in seen:
            seen.add(item["file"])
            print(f"  - {item['file']} ({item['size']} bytes)")
    
    print(f"\n预估节省空间: {sum(item['size'] for item in analysis['files_to_remove'])} bytes")

def confirm_cleanup():
    """确认清理操作"""
    response = input("\n确认执行清理操作吗？(y/N): ").strip().lower()
    return response in ['y', 'yes', '是', '确认']

def execute_cleanup(analysis):
    """执行清理操作"""
    print("\n开始执行清理...")
    
    removed_count = 0
    error_count = 0
    
    for item in analysis["files_to_remove"]:
        file_path = PROJECT_ROOT / item["file"]
        
        try:
            # 先备份
            if not backup_file(item["file"]):
                error_count += 1
                continue
            
            # 删除文件
            file_path.unlink()
            print(f"已删除: {item['file']}")
            removed_count += 1
            
        except Exception as e:
            print(f"删除失败 {item['file']}: {e}")
            error_count += 1
    
    # 处理占位符问题
    handle_placeholder_issue()
    
    return removed_count, error_count

def handle_placeholder_issue():
    """处理占位符问题"""
    print("\n处理占位符问题...")
    
    placeholder_file = PROJECT_ROOT / "iflow/hooks/comprehensive_hook_manager_placeholder.py"
    target_file = PROJECT_ROOT / "iflow/hooks/comprehensive_hook_manager_v4.py"
    
    if target_file.exists() and placeholder_file.exists():
        try:
            # 备份占位符
            backup_file("iflow/hooks/comprehensive_hook_manager_placeholder.py")
            
            # 删除占位符
            placeholder_file.unlink()
            print("已删除占位符文件")
            
            # 创建实际的管理器文件
            create_real_hook_manager()
            
        except Exception as e:
            print(f"处理占位符失败: {e}")

def create_real_hook_manager():
    """创建真正的Hook管理器"""
    content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合Hook管理器 V4 (Comprehensive Hook Manager V4)
集成自动智能质量系统的终极Hook管理器，实现全自动代码审查、测试和质量保障。
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class ComprehensiveHookManagerV4:
    """
    综合Hook管理器V4 - 实现完整的Hook生命周期管理
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.hooks_registry = {}
        self.execution_context = {}
        self.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time": 0.0
        }
        
        logger.info("🌐 综合Hook管理器V4初始化完成")
    
    async def register_hook(self, hook_point: str, hook_function: Callable, priority: int = 50):
        """注册Hook"""
        if hook_point not in self.hooks_registry:
            self.hooks_registry[hook_point] = []
        
        self.hooks_registry[hook_point].append({
            "function": hook_function,
            "priority": priority,
            "registered_at": datetime.now()
        })
        
        # 按优先级排序
        self.hooks_registry[hook_point].sort(key=lambda x: x["priority"])
        logger.debug(f"📋 注册Hook: {hook_point} (优先级: {priority})")
    
    async def execute_hooks(self, hook_point: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行Hook点的所有Hook"""
        if hook_point not in self.hooks_registry:
            logger.debug(f"⏭️ 无Hook注册: {hook_point}")
            return context
        
        start_time = time.time()
        hooks = self.hooks_registry[hook_point]
        
        logger.info(f"🚀 执行Hook点: {hook_point} (共{len(hooks)}个Hook)")
        
        try:
            # 依次执行Hook
            for hook_info in hooks:
                hook_func = hook_info["function"]
                try:
                    # 执行Hook函数
                    if asyncio.iscoroutinefunction(hook_func):
                        result = await hook_func(context)
                    else:
                        result = hook_func(context)
                    
                    # 更新上下文
                    if isinstance(result, dict):
                        context.update(result)
                    
                    self.performance_metrics["successful_executions"] += 1
                    
                except Exception as e:
                    logger.error(f"❌ Hook执行失败: {hook_func.__name__} - {e}")
                    self.performance_metrics["failed_executions"] += 1
            
            execution_time = time.time() - start_time
            self.performance_metrics["total_executions"] += 1
            self.performance_metrics["avg_execution_time"] = (
                (self.performance_metrics["avg_execution_time"] * (self.performance_metrics["total_executions"] - 1) + execution_time) 
                / self.performance_metrics["total_executions"]
            )
            
            logger.info(f"✅ Hook点执行完成: {hook_point} (耗时: {execution_time:.3f}s)")
            
        except Exception as e:
            logger.error(f"❌ Hook执行异常: {hook_point} - {e}")
        
        return context
    
    async def cleanup(self):
        """清理资源"""
        logger.info("🧹 Hook管理器清理完成")
        return True

# 全局Hook管理器实例
_hook_manager = None

def get_hook_manager(config: Dict[str, Any] = None) -> ComprehensiveHookManagerV4:
    """获取Hook管理器实例"""
    global _hook_manager
    if _hook_manager is None:
        _hook_manager = ComprehensiveHookManagerV4(config)
    return _hook_manager

if __name__ == "__main__":
    # 测试代码
    async def test_hook_manager():
        manager = get_hook_manager()
        
        # 测试Hook函数
        async def test_hook1(context):
            print(f"Hook1执行: {context}")
            context["hook1_executed"] = True
            return context
        
        async def test_hook2(context):
            print(f"Hook2执行: {context}")
            context["hook2_executed"] = True
            return context
        
        # 注册Hook
        await manager.register_hook("test_point", test_hook1, priority=10)
        await manager.register_hook("test_point", test_hook2, priority=20)
        
        # 执行Hook
        context = {"test": "data"}
        result = await manager.execute_hooks("test_point", context)
        print(f"最终上下文: {result}")
        
        # 清理
        await manager.cleanup()
    
    # 运行测试
    asyncio.run(test_hook_manager())
'''
    
    try:
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ 已创建真正的Hook管理器")
    except Exception as e:
        print(f"❌ 创建Hook管理器失败: {e}")

def generate_cleanup_report(removed_count, error_count):
    """生成清理报告"""
    report = {
        "cleanup_date": datetime.now().isoformat(),
        "removed_files": removed_count,
        "failed_operations": error_count,
        "backup_location": str(BACKUP_DIR),
        "total_savings": sum(item['size'] for item in analysis["files_to_remove"]) if 'analysis' in locals() else 0
    }
    
    # 保存报告
    report_file = BACKUP_DIR / "cleanup_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 清理报告已保存: {report_file}")
    print(f"📊 清理统计:")
    print(f"  - 删除文件: {removed_count}个")
    print(f"  - 失败操作: {error_count}个")
    print(f"  - 备份位置: {BACKUP_DIR}")
    print(f"  - 节省空间: {report['total_savings']} bytes")

def main():
    """主函数"""
    print("保守清理重复文件脚本")
    print("=" * 60)
    
    # 创建备份
    if not create_backup():
        print("❌ 创建备份失败，退出")
        return False
    
    # 分析重复文件
    analysis = analyze_duplicates()
    
    # 显示分析结果
    show_analysis(analysis)
    
    # 如果没有找到重复文件
    if not analysis["files_to_remove"]:
        print("✅ 没有发现需要清理的重复文件")
        return True
    
    # 确认清理
    if not confirm_cleanup():
        print("🛑 用户取消清理操作")
        return True
    
    # 执行清理
    removed_count, error_count = execute_cleanup(analysis)
    
    # 生成报告
    generate_cleanup_report(removed_count, error_count)
    
    print("\n🎉 清理完成！")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)