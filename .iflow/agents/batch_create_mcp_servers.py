#!/usr/bin/env python3
"""
批量创建MCP服务器脚本
根据settings.json中的配置自动为缺失MCP服务器文件的智能体创建服务器
"""

import os
import json
import logging
from typing import Dict, Any, List

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 智能体配置映射
AGENT_CONFIGS = {
    "adaptive3-thinking": {
        "server_name": "adaptive3_thinking_server",
        "tools": ["multidimensional_thinking", "deep_analysis", "decision_making", "innovation_support"],
        "description": "ADAPTIVE-3思考专家MCP服务器"
    },
    "cluely-assistant": {
        "server_name": "cluely_assistant_server", 
        "tools": ["conversation", "task_assistance", "dialogue_management", "intelligent_response"],
        "description": "Cluely智能助手MCP服务器"
    },
    "code-coverage-analyst": {
        "server_name": "code_coverage_analyst_server",
        "tools": ["coverage_analysis", "code_quality", "test Optimization", "quality_metrics"],
        "description": "代码覆盖率分析师MCP服务器"
    },
    "collaboration-mechanism": {
        "server_name": "collaboration_mechanism_server",
        "tools": ["team_collaboration", "coordination", "communication", "project_management"],
        "description": "协作机制专家MCP服务器"
    },
    "comet-browser-assistant": {
        "server_name": "comet_browser_assistant_server",
        "tools": ["browser_automation", "email_management", "calendar_integration", "web_scraping"],
        "description": "Comet浏览器助手MCP服务器"
    },
    "data-architect": {
        "server_name": "data_architect_server",
        "tools": ["data_modeling", "database_optimization", "big_data_architecture", "data_governance"],
        "description": "数据架构师MCP服务器"
    },
    "devops-engineer": {
        "server_name": "devops_engineer_server",
        "tools": ["ci_cd_pipeline", "containerization", "monitoring", "infrastructure_as_code"],
        "description": "DevOps工程师MCP服务器"
    },
    "fullstack-mentor": {
        "server_name": "fullstack_mentor_server",
        "tools": ["fullstack_development", "frontend_backend", "teaching", "code_guidance"],
        "description": "全栈开发导师MCP服务器"
    },
    "interactive-cli-tool": {
        "server_name": "interactive_cli_tool_server",
        "tools": ["command_execution", "automation", "scripting", "terminal_operations"],
        "description": "交互式命令行工具MCP服务器"
    },
    "it-architect": {
        "server_name": "it_architect_server",
        "tools": ["enterprise_architecture", "integration", "infrastructure_design", "technology_assessment"],
        "description": "IT架构师MCP服务器"
    },
    "live-meeting-co-pilot-cluely": {
        "server_name": "live_meeting_copilot_server",
        "tools": ["meeting_management", "record_transcribe", "collaboration_tools", "schedule_coordination"],
        "description": "实时会议副驾驶MCP服务器"
    },
    "quality-test-engineer": {
        "server_name": "quality_test_engineer_server",
        "tools": ["software_testing", "quality_assurance", "performance_testing", "defect_analysis"],
        "description": "质量测试工程师MCP服务器"
    },
    "security-auditor": {
        "server_name": "security_mcp_server",
        "tools": ["security_scan", "vulnerability_assessment"],
        "description": "安全专用MCP服务器"
    },
    "tech-stack-analyst": {
        "server_name": "tech_stack_analyst_server",
        "tools": ["technology_comparison", "cost_analysis", "performance_evaluation", "recommendation"],
        "description": "技术栈分析师MCP服务器"
    },
    "ui-ux-designer": {
        "server_name": "ui_ux_designer_server",
        "tools": ["user_research", "interface_design", "prototype_creation", "usability_testing"],
        "description": "UI/UX设计专家MCP服务器"
    }
}

def generate_mcp_server_file(agent_name: str, config: Dict[str, Any]) -> str:
    """
    生成MCP服务器文件内容
    
    Args:
        agent_name: 智能体名称
        config: 智能体配置
    
    Returns:
        生成的文件内容
    """
    server_name = config["server_name"]
    tools = config["tools"]
    description = config["description"]
    
    # 生成工具函数
    tools_functions = ""
    for i, tool_name in enumerate(tools):
        func_name = tool_name.replace("-", "_")
        tools_functions += f"""
@tool("{server_name}", "{tool_name}")
@handle_errors
async def {func_name}(
    input_data: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    \"\"\"
    {tool_name}工具
    
    Args:
        input_data: 输入数据
        options: 可选参数
    
    Returns:
        处理结果
    \"\"\"
    logger.info(f"执行{tool_name}工具: {{input_data[:50]}}...")
    
    # 模拟工具执行
    result = {{
        "tool": "{tool_name}",
        "input_processed": input_data,
        "execution_time": datetime.now().isoformat(),
        "result": "工具执行完成",
        "options_applied": options or {{}}
    }}
    
    return create_response(
        success=True,
        data=result,
        message="{tool_name}工具执行完成"
    )
"""
    
    # 生成完整的文件内容
    file_content = f'''#!/usr/bin/env python3
"""
{description}
提供{agent_name}相关的工具和服务
"""

import asyncio
import logging
import sys
from typing import Dict, Any, Optional
from datetime import datetime

# 导入基础框架
sys.path.append('../..')
from agents.mcp_server_framework import tool, get_server, create_response, handle_errors

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# {description}
{agent_name}_server = get_server("{server_name}")

{tools_functions}

# 启动函数
async def main():
    """主函数：启动{description}"""
    logger.info("启动{description}...")
    
    # 注册所有工具到服务器
    logger.info(f"已注册 {{len({agent_name}_server.tools)}} 个工具")
    logger.info(f"可用工具: {{', '.join({agent_name}_server.tools.keys())}}")
    
    print("{description}启动完成")

if __name__ == "__main__":
    asyncio.run(main())
'''
    return file_content

def check_existing_agents() -> List[str]:
    """
    检查已存在的智能体目录
    
    Returns:
        智能体目录列表
    """
    agents_dir = "."
    if not os.path.exists(agents_dir):
        logger.error(f"智能体目录不存在: {agents_dir}")
        return []
    
    # 获取所有子目录（排除文件）
    subdirs = []
    for item in os.listdir(agents_dir):
        item_path = os.path.join(agents_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            subdirs.append(item)
    
    logger.info(f"发现 {len(subdirs)} 个智能体目录: {', '.join(subdirs)}")
    return subdirs

def check_mcp_server_exists(agent_name: str) -> bool:
    """
    检查智能体是否已有MCP服务器文件
    
    Args:
        agent_name: 智能体名称
    
    Returns:
        是否存在MCP服务器文件
    """
    mcp_server_path = f"{agent_name}/mcp_server.py"
    exists = os.path.exists(mcp_server_path)
    if exists:
        logger.info(f"✓ {agent_name}: MCP服务器文件已存在")
    else:
        logger.info(f"✗ {agent_name}: 缺失MCP服务器文件")
    return exists

def create_mcp_server_file(agent_name: str, config: Dict[str, Any]) -> bool:
    """
    创建MCP服务器文件
    
    Args:
        agent_name: 智能体名称
        config: 智能体配置
    
    Returns:
        创建是否成功
    """
    try:
        # 确保目录存在
        os.makedirs(agent_name, exist_ok=True)
        
        # 生成文件内容
        file_content = generate_mcp_server_file(agent_name, config)
        
        # 写入文件
        file_path = f"{agent_name}/mcp_server.py"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(file_content)
        
        logger.info(f"✓ 成功创建 {agent_name} 的MCP服务器文件")
        return True
        
    except Exception as e:
        logger.error(f"✗ 创建 {agent_name} 的MCP服务器文件失败: {str(e)}")
        return False

def main():
    """主函数：批量创建MCP服务器"""
    logger.info("开始批量创建MCP服务器...")
    
    # 检查现有智能体
    existing_agents = check_existing_agents()
    
    # 检查并创建缺失的MCP服务器
    created_count = 0
    skipped_count = 0
    
    for agent_name in existing_agents:
        if agent_name in AGENT_CONFIGS:
            if not check_mcp_server_exists(agent_name):
                config = AGENT_CONFIGS[agent_name]
                if create_mcp_server_file(agent_name, config):
                    created_count += 1
            else:
                skipped_count += 1
        else:
            logger.warning(f"⚠ 未找到 {agent_name} 的配置信息，跳过")
            skipped_count += 1
    
    # 统计结果
    logger.info(f"\\n批量创建完成:")
    logger.info(f"✓ 新创建: {created_count} 个MCP服务器")
    logger.info(f"✓ 已存在: {skipped_count} 个MCP服务器")
    logger.info(f"总处理: {created_count + skipped_count} 个智能体")
    
    if created_count > 0:
        logger.info(f"\\n新创建的MCP服务器:")
        for agent_name in AGENT_CONFIGS:
            if not check_mcp_server_exists(agent_name):
                logger.info(f"  - {agent_name}")

if __name__ == "__main__":
    main()