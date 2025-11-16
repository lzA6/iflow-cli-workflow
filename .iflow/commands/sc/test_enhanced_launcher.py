#!/usr/bin/env python3
"""
增强版 /sc:test 启动脚本
提供简化的命令行接口
"""

import sys
import os
from pathlib import Path

# 添加模块路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from test_enhanced_main import EnhancedSCTestCommand
import asyncio

def main():
    """主函数"""
    print("🚀 增强版 /sc:test 启动器")
    print("=" * 50)
    
    # 获取项目根目录
    project_root = Path.cwd()
    
    # 检查是否在正确的项目目录中
    if not (project_root / ".iflow").exists():
        print("❌ 错误：未找到 .iflow 目录")
        print("请确保在 iFlow CLI 项目根目录中运行此命令")
        sys.exit(1)
    
    print(f"📁 项目根目录: {project_root}")
    print(f"🔧 开始增强版分析...")
    
    # 创建增强版测试实例
    enhanced_test = EnhancedSCTestCommand(str(project_root))
    
    # 运行增强版测试
    try:
        results = asyncio.run(enhanced_test.execute_enhanced_test(
            interactive_mode=True,
            force_ai_awareness=True,
            enable_deep_analysis=True,
            generate_optimization_report=True,
            compare_structures=True
        ))
        
        print("\n" + "=" * 50)
        print("🎉 增强版 /sc:test 分析完成！")
        print("=" * 50)
        
        # 显示关键结果
        if results:
            print(f"📊 分析状态: {results['executive_summary']['overall_status']}")
            
            # 显示结论
            if results.get('conclusions'):
                print(f"\n🎯 主要结论:")
                for conclusion in results['conclusions'][:3]:
                    print(f"  • {conclusion}")
            
            # 显示下一步行动
            if results.get('next_steps'):
                print(f"\n📋 下一步行动:")
                for step in results['next_steps'][:3]:
                    print(f"  • {step}")
        
        print(f"\n📄 详细报告已保存到 reports/ 目录")
        print("💡 使用 --no-interactive 参数可跳过交互模式")
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，分析已停止")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()