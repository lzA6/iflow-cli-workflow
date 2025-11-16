#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ARP智能提示词优化命令 (ARP Intelligent Prompt Optimization Command)
======================================================================

为ARP系统添加智能提示词优化功能的命令行接口：
- 🎯 交互式提示词优化
- 📊 实时用户反馈收集
- 📈 个性化优化建议
- 💾 本地数据持久化
- 🧠 自动学习用户偏好
- 🔄 多模式切换支持

使用方式：
- 直接输入提示词开始优化
- 使用数字键选择优化模式
- 系统自动记忆用户偏好

作者: iFlow架构团队
版本: 1.0.0
日期: 2025-11-17
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# 项目根路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.intelligent_prompt_optimizer import (
    IntelligentPromptOptimizer,
    OptimizationMode,
    get_prompt_optimizer,
    optimize_user_prompt
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARPPromptOptimizationCommand:
    """ARP提示词优化命令处理器"""
    
    def __init__(self):
        self.optimizer = get_prompt_optimizer()
        self.current_user_id = "default_user"
        self.current_session = None
        self.optimization_cache = {}
        
    def print_welcome(self):
        """打印欢迎信息"""
        print("""
🧠 ARP智能提示词优化器 V1.0
===============================

✨ 功能特点：
• 自动优化您的提示词，让AI更懂您的需求
• 5种优化模式：标准、专业、小白、AI格式、重新优化
• 智能学习您的偏好，越用越懂您
• 所有数据本地保存，隐私安全

📍 数据存储位置：{}
📝 使用说明：直接输入您的提示词开始优化

🚀 让我们开始吧！请输入您想要优化的提示词：
        """.format(self.optimizer.data_dir))
    
    def print_optimization_result(self, result):
        """打印优化结果"""
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
        
        print("\n🔄 下一步操作：")
        for i, step in enumerate(result.next_steps, 1):
            print(f"  {step}")
        
        print("\n🎯 请输入数字选择操作（1-5），或输入新提示词重新开始：")
    
    def print_user_statistics(self, user_id: str):
        """打印用户统计信息"""
        stats = self.optimizer.get_user_statistics(user_id)
        
        print(f"\n📊 用户统计 - {user_id}")
        print("-" * 40)
        print(f"💬 总交互次数：{stats['total_interactions']}")
        print(f"✅ 接受率：{stats['acceptance_rate']:.1f}%")
        print(f"⭐ 平均满意度：{stats['average_satisfaction']:.1f}/5.0")
        print(f"🎓 专业水平：{stats['expertise_level']}")
        
        if stats['preferred_modes']:
            print(f"🎯 偏好模式：")
            for mode, count in stats['preferred_modes']:
                print(f"    • {mode}: {count}次")
        
        if stats['satisfaction_trend']:
            print(f"📈 最近满意度趋势：{stats['satisfaction_trend']}")
    
    async def handle_optimization_request(self, user_input: str):
        """处理优化请求"""
        # 检查是否是模式切换命令
        if user_input.isdigit() and len(user_input) == 1:
            await self.handle_mode_selection(int(user_input))
            return
        
        # 检查是否是统计命令
        if user_input.lower() in ['stats', '统计', '我的数据']:
            self.print_user_statistics(self.current_user_id)
            print("\n🎯 请输入新的提示词继续优化：")
            return
        
        # 检查是否是帮助命令
        if user_input.lower() in ['help', '帮助', '?']:
            self.print_help()
            return
        
        # 检查是否是导出命令
        if user_input.lower() in ['export', '导出', '备份']:
            export_path = self.optimizer.export_user_data(self.current_user_id)
            print(f"📁 用户数据已导出到：{export_path}")
            print("\n🎯 请输入新的提示词继续优化：")
            return
        
        # 处理为新的优化请求
        await self.optimize_prompt(user_input)
    
    async def optimize_prompt(self, prompt: str, mode: OptimizationMode = OptimizationMode.STANDARD):
        """优化提示词"""
        print(f"\n🔄 正在优化提示词（模式：{mode.value}）...")
        
        result = await self.optimizer.optimize_prompt(self.current_user_id, prompt, mode)
        
        if result.success:
            self.optimization_cache['last_result'] = result
            self.optimization_cache['original_prompt'] = prompt
            self.print_optimization_result(result)
        else:
            print(f"❌ 优化失败：{result.reasoning}")
            print("🎯 请重新输入提示词：")
    
    async def handle_mode_selection(self, choice: int):
        """处理模式选择"""
        if 'last_result' not in self.optimization_cache:
            print("❌ 没有可重新优化的内容，请先输入提示词")
            print("\n🎯 请输入新的提示词：")
            return
        
        original_prompt = self.optimization_cache['original_prompt']
        
        if choice == 1:
            # 确认使用优化后的提示词
            result = self.optimization_cache['last_result']
            self.optimizer.record_feedback(
                record_id=self.optimizer.optimization_history[-1].record_id,
                user_feedback=5,
                user_accepted=True
            )
            print(f"✅ 已确认使用优化后的提示词")
            print(f"📝 优化后的提示词：{result.optimized_prompt}")
            print("\n🎯 感谢使用！您可以输入新的提示词继续优化：")
            
        elif choice == 2:
            # 重新优化
            print("🔄 重新优化当前提示词...")
            await self.optimize_prompt(original_prompt, OptimizationMode.REOPTIMIZE)
            
        elif choice == 3:
            # 专业方向优化
            print("🎓 切换到专业方向优化...")
            await self.optimize_prompt(original_prompt, OptimizationMode.PROFESSIONAL)
            
        elif choice == 4:
            # 小白友好模式
            print("🌱 切换到小白友好模式...")
            await self.optimize_prompt(original_prompt, OptimizationMode.BEGINNER)
            
        elif choice == 5:
            # AI友好格式
            print("🤖 切换到AI友好格式...")
            await self.optimize_prompt(original_prompt, OptimizationMode.AI_FORMAT)
            
        else:
            print("❌ 无效选择，请输入1-5的数字")
            self.print_optimization_result(self.optimization_cache['last_result'])
    
    def print_help(self):
        """打印帮助信息"""
        help_text = """
📖 帮助信息
==========

🎯 基本使用：
• 直接输入提示词开始优化
• 系统会自动优化并提供选择

🔄 模式说明：
1. 确认使用：接受当前优化结果
2. 重新优化：基于反馈重新优化
3. 专业方向：添加专业术语和技术细节
4. 小白友好：简化表达，通俗易懂
5. AI格式：结构化格式，AI更易理解

📊 其他命令：
• stats/统计：查看个人使用统计
• export/导出：导出个人数据
• help/帮助：显示此帮助信息

💾 数据隐私：
• 所有数据存储在本地：{}
• 不会被上传到任何服务器
• 您可以随时导出或清理数据

🎓 智能学习：
• 系统会学习您的偏好
• 越用越懂您的需求
• 自动调整优化策略
        """.format(self.optimizer.data_dir)
        print(help_text)
        print("\n🎯 请输入提示词开始优化：")
    
    async def start_interactive_session(self, user_id: Optional[str] = None):
        """启动交互式会话"""
        if user_id:
            self.current_user_id = user_id
            # 确保用户存在
            self.optimizer.get_or_create_user(user_id)
        
        self.print_welcome()
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("\n👋 感谢使用ARP智能提示词优化器！")
                    break
                
                await self.handle_optimization_request(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 感谢使用ARP智能提示词优化器！")
                break
            except Exception as e:
                logger.error(f"处理用户输入时出错：{e}")
                print("❌ 处理输入时出现错误，请重试")

# 全局命令实例
_global_command: Optional[ARPPromptOptimizationCommand] = None

def get_optimization_command() -> ARPPromptOptimizationCommand:
    """获取全局优化命令实例"""
    global _global_command
    if _global_command is None:
        _global_command = ARPPromptOptimizationCommand()
    return _global_command

async def start_prompt_optimization(user_id: Optional[str] = None):
    """启动提示词优化会话"""
    command = get_optimization_command()
    await command.start_interactive_session(user_id)

if __name__ == "__main__":
    # 直接运行启动交互式会话
    asyncio.run(start_prompt_optimization())
