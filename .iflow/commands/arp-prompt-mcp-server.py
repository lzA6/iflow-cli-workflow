#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARP Prompt Optimizer MCP Server
==============================

MCP服务器，提供ARP智能提示词优化功能集成

作者: iFlow架构团队
版本: 1.0.0
日期: 2025-11-17
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# 添加.iflow路径
iflow_path = project_root / ".iflow"
sys.path.insert(0, str(iflow_path))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    from core.intelligent_prompt_optimizer import (
        OptimizationMode,
        optimize_user_prompt,
        get_prompt_optimizer
    )
except ImportError as e:
    print(f"❌ 导入错误：{e}")
    print("请确保安装了MCP相关依赖")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arp-prompt-mcp-server")

# 创建MCP服务器
server = Server("arp-prompt-optimizer")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """列出可用工具"""
    return [
        Tool(
            name="optimize_prompt",
            description="优化提示词，支持多种模式",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "要优化的提示词"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["standard", "professional", "beginner", "ai_format", "reoptimize"],
                        "default": "standard",
                        "description": "优化模式"
                    },
                    "user_id": {
                        "type": "string",
                        "default": "default_user",
                        "description": "用户ID"
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="get_user_stats",
            description="获取用户统计信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "default": "default_user",
                        "description": "用户ID"
                    }
                }
            }
        ),
        Tool(
            name="export_user_data",
            description="导出用户数据",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "default": "default_user",
                        "description": "用户ID"
                    }
                }
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """处理工具调用"""
    try:
        if name == "optimize_prompt":
            prompt = arguments.get("prompt", "")
            mode = arguments.get("mode", "standard")
            user_id = arguments.get("user_id", "default_user")
            
            if not prompt:
                return [TextContent(
                    type="text",
                    text="❌ 错误：请提供要优化的提示词"
                )]
            
            # 优化提示词
            optimization_mode = OptimizationMode(mode)
            result = await optimize_user_prompt(user_id, prompt, optimization_mode)
            
            if result.success:
                response = f"""🎯 提示词优化结果
=================

✅ 优化模式：{result.optimization_mode.value}
📊 置信度：{result.confidence:.2f}
💡 优化说明：{result.reasoning}

📝 优化后的提示词：
------------------------
{result.optimized_prompt}
------------------------

💡 优化建议：
"""
                if result.suggestions:
                    for i, suggestion in enumerate(result.suggestions, 1):
                        response += f"\n  {i}. {suggestion}"
                else:
                    response += "\n  暂无特别建议"
                
                return [TextContent(type="text", text=response)]
            else:
                return [TextContent(
                    type="text",
                    text=f"❌ 优化失败：{result.reasoning}"
                )]
        
        elif name == "get_user_stats":
            user_id = arguments.get("user_id", "default_user")
            
            try:
                optimizer = get_prompt_optimizer()
                stats = optimizer.get_user_statistics(user_id)
                
                response = f"""📊 用户统计信息 - {user_id}
========================

💬 总交互次数：{stats['total_interactions']}
✅ 接受率：{stats['acceptance_rate']:.1f}%
⭐ 平均满意度：{stats['average_satisfaction']:.1f}/5.0
🎓 专业水平：{stats['expertise_level']}
"""
                
                if stats['preferred_modes']:
                    response += "\n🎯 偏好模式：\n"
                    for mode, count in stats['preferred_modes']:
                        response += f"  • {mode}: {count}次\n"
                
                if stats['satisfaction_trend']:
                    response += f"\n📈 最近满意度趋势：{stats['satisfaction_trend']}\n"
                
                response += f"\n📁 数据存储位置：{optimizer.data_dir}"
                
                return [TextContent(type="text", text=response)]
                
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"❌ 获取统计信息失败：{e}"
                )]
        
        elif name == "export_user_data":
            user_id = arguments.get("user_id", "default_user")
            
            try:
                optimizer = get_prompt_optimizer()
                export_path = optimizer.export_user_data(user_id)
                
                response = f"""📁 用户数据导出成功
==================

用户ID：{user_id}
导出路径：{export_path}

数据包含：
- 用户画像信息
- 优化历史记录
- 偏好设置
- 训练数据

隐私保护：所有数据仅存储在本地
"""
                
                return [TextContent(type="text", text=response)]
                
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"❌ 导出用户数据失败：{e}"
                )]
        
        else:
            return [TextContent(
                type="text",
                text=f"❌ 未知工具：{name}"
            )]
            
    except Exception as e:
        logger.error(f"工具调用错误：{e}")
        return [TextContent(
            type="text",
            text=f"❌ 执行错误：{e}"
        )]

async def main():
    """主函数"""
    logger.info("🧠 ARP提示词优化器MCP服务器启动")
    logger.info("📊 提供智能提示词优化功能")
    
    # 使用stdio运行服务器
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())