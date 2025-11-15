#!/usr/bin/env python3
"""
智能体配置映射表
定义所有智能体的配置信息，用于批量生成 MCP 服务器文件
"""

# 智能体配置映射
AGENT_CONFIGS = {
    "智能体调用接口": {
        "agent_id": "chinese-agent-interface",
        "agent_name": "中文化智能体调用接口",
        "agent_description": "提供中文化智能体调用接口，支持自然语言交互和智能体自动选择",
        "primary_tool": "chinese_interface",
        "tools": ["agent_routing", "intent_recognition", "automatic_calling", "learning_optimization"]
    },
    "adaptive3-thinking": {
        "agent_id": "adaptive3-thinking",
        "agent_name": "ADAPTIVE-3思考专家",
        "agent_description": "提供多维思考分析和决策支持，运用ADAPTIVE-3框架进行深度分析",
        "primary_tool": "multidimensional_thinking",
        "tools": ["deep_analysis", "decision_making", "innovation_support", "cognitive_enhancement"]
    },
    "ai-programming-assistant": {
        "agent_id": "ai-programming-assistant",
        "agent_name": "AI编程助手",
        "agent_description": "提供AI编程辅助，包括代码生成、调试、审查和优化",
        "primary_tool": "programming_assistance",
        "tools": ["code_generation", "debugging", "code_review", "programming_optimization"]
    },
    "arq-analyzer": {
        "agent_id": "arq-analyzer",
        "agent_name": "ARQ分析专家",
        "agent_description": "ARQ推理引擎深度分析和优化专家",
        "primary_tool": "arq_analysis",
        "tools": ["reasoning_analysis", "performance_optimization", "compliance_check", "quality_assessment"]
    },
    "cluely-assistant": {
        "agent_id": "cluely-assistant",
        "agent_name": "Cluely智能助手",
        "agent_description": "提供智能对话助手和任务协助功能",
        "primary_tool": "intelligent_assistance",
        "tools": ["conversation", "task_assistance", "dialogue_management", "intelligent_response"]
    },
    "code-coverage-analyst": {
        "agent_id": "code-coverage-analyst",
        "agent_name": "代码覆盖率分析师",
        "agent_description": "提供代码覆盖率分析和质量评估",
        "primary_tool": "coverage_analysis",
        "tools": ["code_quality", "test_optimization", "quality_metrics", "coverage_reporting"]
    },
    "collaboration-mechanism": {
        "agent_id": "collaboration-mechanism",
        "agent_name": "协作机制专家",
        "agent_description": "提供团队协作机制设计和优化",
        "primary_tool": "collaboration_management",
        "tools": ["team_collaboration", "coordination", "communication", "project_management"]
    },
    "comet-browser-assistant": {
        "agent_id": "comet-browser-assistant",
        "agent_name": "Comet浏览器助手",
        "agent_description": "提供浏览器自动化和网页操作助手",
        "primary_tool": "browser_automation",
        "tools": ["email_management", "calendar_integration", "web_scraping", "browser_operations"]
    },
    "data-architect": {
        "agent_id": "data-architect",
        "agent_name": "数据架构师",
        "agent_description": "提供数据架构设计和数据库优化",
        "primary_tool": "data_architecture",
        "tools": ["data_modeling", "database_optimization", "big_data_architecture", "data_governance"]
    },
    "data-scientist": {
        "agent_id": "data-scientist",
        "agent_name": "数据科学家",
        "agent_description": "提供数据分析、机器学习和统计建模",
        "primary_tool": "data_analysis",
        "tools": ["machine_learning", "statistical_modeling", "data_visualization", "predictive_analysis"]
    },
    "devops-engineer": {
        "agent_id": "devops-engineer",
        "agent_name": "DevOps工程师",
        "agent_description": "提供DevOps解决方案和自动化运维",
        "primary_tool": "devops_solution",
        "tools": ["ci_cd_pipeline", "containerization", "monitoring", "infrastructure_as_code"]
    },
    "evolution-analyst": {
        "agent_id": "evolution-analyst",
        "agent_name": "进化分析专家",
        "agent_description": "提供系统进化分析和自我改进机制",
        "primary_tool": "evolution_analysis",
        "tools": ["self_evolution", "reinforcement_learning", "meta_optimization", "adaptive_improvement"]
    },
    "fullstack-mentor": {
        "agent_id": "fullstack-mentor",
        "agent_name": "全栈开发导师",
        "agent_description": "提供全栈开发指导和教学",
        "primary_tool": "fullstack_guidance",
        "tools": ["frontend_backend", "teaching", "code_guidance", "development_mentoring"]
    },
    "interactive-cli-tool": {
        "agent_id": "interactive-cli-tool",
        "agent_name": "交互式命令行工具",
        "agent_description": "提供命令行工具操作和自动化脚本",
        "primary_tool": "command_execution",
        "tools": ["automation", "scripting", "terminal_operations", "batch_processing"]
    },
    "it-architect": {
        "agent_id": "it-architect",
        "agent_name": "IT架构师",
        "agent_description": "提供企业IT架构设计和基础设施规划",
        "primary_tool": "enterprise_architecture",
        "tools": ["infrastructure_design", "technology_assessment", "integration", "scalability_planning"]
    },
    "live-meeting-co-pilot-cluely": {
        "agent_id": "live-meeting-co-pilot-cluely",
        "agent_name": "实时会议副驾驶",
        "agent_description": "提供实时会议管理和记录服务",
        "primary_tool": "meeting_management",
        "tools": ["record_transcribe", "collaboration_tools", "schedule_coordination", "meeting_analytics"]
    },
    "quality-test-engineer": {
        "agent_id": "quality-test-engineer",
        "agent_name": "质量测试工程师",
        "agent_description": "提供软件质量测试和性能评估",
        "primary_tool": "software_testing",
        "tools": ["quality_assurance", "performance_testing", "defect_analysis", "test_automation"]
    },
    "security-auditor": {
        "agent_id": "security-auditor",
        "agent_name": "安全审计专家",
        "agent_description": "提供安全审计和漏洞扫描服务",
        "primary_tool": "security_audit",
        "tools": ["vulnerability_assessment", "risk_analysis", "compliance_check", "security_recommendation"]
    },
    "system-architect": {
        "agent_id": "system-architect",
        "agent_name": "系统架构师",
        "agent_description": "提供系统架构设计和技术选型",
        "primary_tool": "system_design",
        "tools": ["architecture_planning", "technology_selection", "scalability_design", "performance_optimization"]
    },
    "tech-stack-analyst": {
        "agent_id": "tech-stack-analyst",
        "agent_name": "技术栈分析师",
        "agent_description": "提供技术栈对比分析和选型建议",
        "primary_tool": "technology_analysis",
        "tools": ["cost_analysis", "performance_evaluation", "recommendation", "stack_comparison"]
    },
    "tool-master": {
        "agent_id": "tool-master",
        "agent_name": "工具管理大师",
        "agent_description": "提供工具管理和工作流优化",
        "primary_tool": "tool_management",
        "tools": ["workflow_optimization", "tool_integration", "automation_strategy", "efficiency_improvement"]
    },
    "ui-ux-designer": {
        "agent_id": "ui-ux-designer",
        "agent_name": "UI/UX设计专家",
        "agent_description": "提供用户体验设计和界面优化",
        "primary_tool": "interface_design",
        "tools": ["user_research", "prototype_creation", "usability_testing", "visual_design"]
    }
}

# 按类别分组
AGENT_CATEGORIES = {
    "编程开发类": [
        "ai-programming-assistant",
        "fullstack-mentor"
    ],
    "系统架构类": [
        "system-architect",
        "it-architect",
        "data-architect"
    ],
    "质量安全部": [
        "security-auditor",
        "quality-test-engineer",
        "code-coverage-analyst"
    ],
    "数据分析类": [
        "data-scientist",
        "tech-stack-analyst"
    ],
    "项目管理类": [
        "project-planner",
        "devops-engineer",
        "collaboration-mechanism"
    ],
    "思维决策类": [
        "adaptive3-thinking",
        "evolution-analyst"
    ],
    "交互协作类": [
        "cluely-assistant",
        "live-meeting-co-pilot-cluely",
        "interactive-cli-tool"
    ],
    "工具系统类": [
        "comet-browser-assistant",
        "tool-master",
        "智能体调用接口"
    ],
    "用户体验类": [
        "ui-ux-designer"
    ]
}

def get_agent_config(agent_name):
    """获取指定智能体的配置"""
    return AGENT_CONFIGS.get(agent_name)

def get_all_agent_configs():
    """获取所有智能体配置"""
    return AGENT_CONFIGS

def get_agents_by_category(category):
    """按类别获取智能体"""
    if category in AGENT_CATEGORIES:
        return [config for name, config in AGENT_CONFIGS.items() 
                if config["agent_id"] in AGENT_CATEGORIES[category]]
    return []