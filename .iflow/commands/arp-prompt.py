#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARP提示词优化命令 - 统一入口
========================

这是/arp-prompt命令的统一入口点，提供智能提示词优化功能

使用方式：
- /arp-prompt "你的提示词" - 直接优化提示词
- /arp-prompt --mode professional "你的提示词" - 使用专业模式
- /arp-prompt --interactive - 启动交互式会话
- /arp-prompt --stats - 查看用户统计

作者: iFlow架构团队
版本: 1.0.0
日期: 2025-11-17
"""

import sys
import os
import time
import argparse
import asyncio
from pathlib import Path

try:
    # 添加项目路径
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    sys.path.insert(0, str(project_root))
    
    # 添加.iflow路径
    iflow_path = project_root / ".iflow"
    sys.path.insert(0, str(iflow_path))
    
    # 导入优化器
    from core.intelligent_prompt_optimizer import (
        OptimizationMode,
        optimize_user_prompt,
        get_prompt_optimizer
    )
except ImportError as e:
    print(f"❌ 导入错误：{e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

def print_banner():
    """打印横幅"""
    print("""
🧠 ARP智能提示词优化器 V1.0
===============================
✨ 智能优化您的提示词，让AI更懂您的需求
🎯 5种优化模式，个性化适配
💾 本地数据存储，隐私安全
📈 越用越懂您，持续学习
    """)

async def optimize_single_prompt(prompt: str, mode: str = "standard", user_id: str = "default_user"):
    """优化单个提示词"""
    print(f"🔄 正在优化提示词（模式：{mode}）...")
    
    try:
        optimization_mode = OptimizationMode(mode)
        result = await optimize_user_prompt(user_id, prompt, optimization_mode)
        
        if result.success:
            print("\n" + "="*60)
            print("🎯 优化结果")
            print("="*60)
            print(f"✅ 优化模式：{result.optimization_mode.value}")
            print(f"📊 置信度：{result.confidence:.2f}")
            print(f"💡 优化 reasoning：{result.reasoning}")
            
            print("\n📝 优化后的提示词：")
            print("-" * 40)
            print(result.optimized_prompt)
            print("-" * 40)
            
            if result.suggestions:
                print("\n💡 建议：")
                for i, suggestion in enumerate(result.suggestions, 1):
                    print(f"  {i}. {suggestion}")
            
            # 处理时间信息（当前版本暂不支持）
            # print(f"\n⏱️  处理时间：{result.processing_time:.3f}秒")
            
            # 记录用户确认
            print("\n🎯 优化完成！您可以直接使用这个提示词。")
            
        else:
            print(f"❌ 优化失败：{result.reasoning}")
            
    except Exception as e:
        print(f"❌ 优化过程中出现错误：{e}")

async def start_interactive_session(user_id: str = "default_user"):
    """启动交互式会话"""
    print_banner()
    print("🌊 进入交互式模式，输入 'quit' 或 '退出' 结束会话")
    print("🎯 请输入您想要优化的提示词：")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("\n👋 感谢使用ARP智能提示词优化器！")
                break
            
            await optimize_single_prompt(user_input, "standard", user_id)
            
        except KeyboardInterrupt:
            print("\n\n👋 感谢使用ARP智能提示词优化器！")
            break
        except Exception as e:
            print(f"❌ 处理输入时出现错误：{e}")

def show_user_stats(user_id: str = "default_user"):
    """显示用户统计"""
    optimizer = get_prompt_optimizer()
    stats = optimizer.get_user_statistics(user_id)
    
    print(f"\n📊 用户统计 - {user_id}")
    print("=" * 50)
    print(f"💬 总交互次数：{stats['total_interactions']}")
    print(f"✅ 接受率：{stats['acceptance_rate']:.1f}%")
    print(f"⭐ 平均满意度：{stats['average_satisfaction']:.1f}/5.0")
    print(f"🎓 专业水平：{stats['expertise_level']}")
    
    if stats['preferred_modes']:
        print(f"\n🎯 偏好模式：")
        for mode, count in stats['preferred_modes']:
            print(f"    • {mode}: {count}次")
    
    if stats['satisfaction_trend']:
        print(f"\n📈 最近满意度趋势：{stats['satisfaction_trend']}")
    
    print(f"\n📁 数据存储位置：{optimizer.data_dir}")

def export_user_data(user_id: str = "default_user"):
    """导出用户数据"""
    optimizer = get_prompt_optimizer()
    export_path = optimizer.export_user_data(user_id)
    print(f"📁 用户数据已导出到：{export_path}")

def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="ARP智能提示词优化器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  arp-prompt "帮我写一个Python函数"
  arp-prompt --mode professional "解释机器学习算法"
  arp-prompt --mode beginner "什么是区块链"
  arp-prompt --interactive
  arp-prompt --stats
  arp-prompt --export --user-id my_user

优化模式：
  standard      - 标准优化（默认）
  professional  - 专业方向，添加术语和技术细节
  beginner      - 小白友好，通俗易懂
  ai_format     - AI友好格式，结构化提示词
  reoptimize    - 重新优化，基于反馈改进
        """
    )
    
    parser.add_argument("prompt", nargs="*", help="要优化的提示词")
    parser.add_argument("--mode", 
                       choices=["standard", "professional", "beginner", "ai_format", "reoptimize"],
                       default="standard", 
                       help="优化模式（默认：standard）")
    parser.add_argument("--user-id", default="default_user", help="用户ID（默认：default_user）")
    parser.add_argument("--interactive", "-i", action="store_true", help="启动交互式会话")
    parser.add_argument("--stats", "-s", action="store_true", help="显示用户统计信息")
    parser.add_argument("--export", "-e", action="store_true", help="导出用户数据")
    parser.add_argument("--batch", "-b", action="store_true", help="批量模式")
    parser.add_argument("--version", "-v", action="version", version="ARP智能提示词优化器 V1.0")
    
    args = parser.parse_args()
    
    # 处理特殊命令
    if args.stats:
        show_user_stats(args.user_id)
        return
    
    if args.export:
        export_user_data(args.user_id)
        return
    
    if args.interactive:
        asyncio.run(start_interactive_session(args.user_id))
        return
    
    # 处理提示词优化
    if args.prompt:
        prompt = " ".join(args.prompt)
        if args.batch:
            # 批量模式
            prompts = prompt.split("|")
            print(f"🚀 批量优化模式，共 {len(prompts)} 个提示词")
            for i, p in enumerate(prompts, 1):
                print(f"\n[{i}/{len(prompts)}] 优化：{p}")
                asyncio.run(optimize_single_prompt(p.strip(), args.mode, args.user_id))
        else:
            # 单个提示词
            asyncio.run(optimize_single_prompt(prompt, args.mode, args.user_id))
    else:
        # 没有提示词，显示帮助
        print_banner()
        print("🎯 请提供要优化的提示词，或使用 --interactive 启动交互式模式")
        print("\n💡 使用示例：")
        print("  arp-prompt \"帮我写代码\"")
        print("  arp-prompt --mode professional \"机器学习算法\"")
        print("  arp-prompt --interactive")
        print("  arp-prompt --stats")

if __name__ == "__main__":
    main()