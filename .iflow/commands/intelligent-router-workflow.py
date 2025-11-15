#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 智能体自动识别和调用系统工作流
Intelligent Agent Router Workflow

专门用于智能体自动识别、意图分析和智能路由，提供最佳的智能体选择和调用服务。

作者: AI架构师团队
版本: 1.0.0
日期: 2025-11-14
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入相关模块
try:
    from .core.intelligent_context_manager import IntelligentContextManager
    from .core.agent_lifecycle_manager_v2 import AgentLifecycleManager
    from .tools.intelligent_dashboard import IntelligentDashboard
except ImportError as e:
    logging.error(f"无法导入依赖模块: {e}")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RoutingConfig:
    """路由配置"""
    auto_mode: bool = True
    manual_mode: bool = False
    debug_mode: bool = False
    confidence_threshold: float = 0.7
    max_recommendations: int = 3

@dataclass
class IntentAnalysis:
    """意图分析结果"""
    primary_intent: str
    confidence: float
    keywords: List[str]
    entities: List[Dict[str, Any]]
    explanation: str

@dataclass
class AgentRecommendation:
    """智能体推荐"""
    agent_id: str
    name: str
    confidence: float
    reason: str
    parameters: Dict[str, Any]

class IntelligentRouterWorkflow:
    """智能体路由工作流"""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        
        # 智能体映射表
        self.agent_mapping = {
            # 架构设计类
            "系统架构": ["system-architect", "系统架构师", 0.9],
            "设计": ["system-architect", "系统架构师", 0.8],
            "架构": ["system-architect", "系统架构师", 0.9],
            "技术架构": ["system-architect", "系统架构师", 0.9],
            "IT架构": ["it-architect", "IT架构师", 0.9],
            "企业架构": ["it-architect", "IT架构师", 0.9],
            "集成": ["it-architect", "IT架构师", 0.8],
            
            # 编程开发类
            "编程": ["ai-programming-assistant", "AI编程助手", 0.9],
            "开发": ["ai-programming-assistant", "AI编程助手", 0.9],
            "代码": ["ai-programming-assistant", "AI编程助手", 0.8],
            "写代码": ["ai-programming-assistant", "AI编程助手", 0.95],
            "调试": ["ai-programming-assistant", "AI编程助手", 0.9],
            "编程": ["fullstack-mentor", "全栈开发导师", 0.8],
            "教学": ["fullstack-mentor", "全栈开发导师", 0.8],
            
            # 项目管理类
            "项目": ["project-planner", "项目规划专家", 0.9],
            "规划": ["project-planner", "项目规划专家", 0.9],
            "管理": ["project-planner", "项目规划专家", 0.8],
            "需求": ["project-planner", "项目规划专家", 0.8],
            "风险管理": ["project-planner", "项目规划专家", 0.9],
            
            # 质量测试类
            "测试": ["quality-test-engineer", "质量测试工程师", 0.9],
            "质量": ["quality-test-engineer", "质量测试工程师", 0.9],
            "QA": ["quality-test-engineer", "质量测试工程师", 0.8],
            "功能测试": ["quality-test-engineer", "质量测试工程师", 0.9],
            "代码测试": ["code-coverage-analyst", "代码覆盖率分析师", 0.9],
            "覆盖率": ["code-coverage-analyst", "代码覆盖率分析师", 0.95],
            
            # 安全分析类
            "安全": ["security-auditor", "安全审计专家", 0.95],
            "漏洞": ["security-auditor", "安全审计专家", 0.9],
            "风险": ["security-auditor", "安全审计专家", 0.9],
            "审计": ["security-auditor", "安全审计专家", 0.9],
            
            # 数据分析类
            "数据": ["data-scientist", "数据科学家", 0.9],
            "分析": ["data-scientist", "数据科学家", 0.8],
            "统计": ["data-scientist", "数据科学家", 0.8],
            "机器学习": ["data-scientist", "数据科学家", 0.9],
            "数据架构": ["data-architect", "数据架构师", 0.95],
            "数据库": ["data-architect", "数据架构师", 0.9],
            
            # 思维决策类
            "思考": ["adaptive3-thinking", "ADAPTIVE-3思考专家", 0.9],
            "决策": ["adaptive3-thinking", "ADAPTIVE-3思考专家", 0.9],
            "创新": ["adaptive3-thinking", "ADAPTIVE-3思考专家", 0.9],
            "分析": ["adaptive3-thinking", "ADAPTIVE-3思考专家", 0.8],
            
            # 协作沟通类
            "协作": ["collaboration-mechanism", "协作机制专家", 0.9],
            "合作": ["collaboration-mechanism", "协作机制专家", 0.8],
            "团队": ["collaboration-mechanism", "协作机制专家", 0.8],
            "协调": ["collaboration-mechanism", "协作机制专家", 0.8],
            "会议": ["live-meeting-co-pilot-cluely", "实时会议副驾驶", 0.9],
            "记录": ["live-meeting-co-pilot-cluely", "实时会议副驾驶", 0.8],
            "聊天": ["cluely-assistant", "Cluely智能助手", 0.9],
            "对话": ["cluely-assistant", "Cluely智能助手", 0.9],
            
            # 工具系统类
            "命令": ["interactive-cli-tool", "交互式命令行工具", 0.9],
            "CLI": ["interactive-cli-tool", "交互式命令行工具", 0.9],
            "终端": ["interactive-cli-tool", "交互式命令行工具", 0.8],
            "自动化": ["interactive-cli-tool", "交互式命令行工具", 0.8],
            "浏览器": ["comet-browser-assistant", "Comet浏览器助手", 0.9],
            "网页": ["comet-browser-assistant", "Comet浏览器助手", 0.8],
            "邮件": ["comet-browser-assistant", "Comet浏览器助手", 0.8],
            "抓取": ["comet-browser-assistant", "Comet浏览器助手", 0.9],
            
            # ARQ推理类
            "ARQ": ["arq-analyzer", "ARQ分析专家", 0.95],
            "推理": ["arq-analyzer", "ARQ分析专家", 0.9],
            "逻辑": ["arq-analyzer", "ARQ分析专家", 0.9],
            "分析": ["arq-analyzer", "ARQ分析专家", 0.8],
            
            # DevOps类
            "DevOps": ["devops-engineer", "DevOps工程师", 0.95],
            "部署": ["devops-engineer", "DevOps工程师", 0.9],
            "运维": ["devops-engineer", "DevOps工程师", 0.9],
            "CI/CD": ["devops-engineer", "DevOps工程师", 0.9],
            
            # UI/UX类
            "设计": ["ui-ux-designer", "UI/UX设计专家", 0.8],
            "界面": ["ui-ux-designer", "UI/UX设计专家", 0.9],
            "用户体验": ["ui-ux-designer", "UI/UX设计专家", 0.95],
            "原型": ["ui-ux-designer", "UI/UX设计专家", 0.8],
            
            # 中文指令
            "中文": ["chinese-commands", "中文指令系统", 0.95],
            "指令": ["chinese-commands", "中文指令系统", 0.9],
            "交互": ["chinese-commands", "中文指令系统", 0.9],
            "语言": ["chinese-commands", "中文指令系统", 0.9],
        }
        
        # 复杂意图模式
        self.complex_patterns = [
            (r"(设计|架构|系统)", ["system-architect"], "系统设计相关"),
            (r"(编程|开发|写程序|编码)", ["ai-programming-assistant"], "编程开发相关"),
            (r"(测试|质量|QA|bug)", ["quality-test-engineer"], "质量测试相关"),
            (r"(安全|漏洞|风险|防护)", ["security-auditor"], "安全分析相关"),
            (r"(数据|分析|统计|处理)", ["data-scientist"], "数据分析相关"),
            (r"(项目|管理|规划|计划)", ["project-planner"], "项目管理相关"),
            (r"(DevOps|部署|运维|发布)", ["devops-engineer"], "DevOps相关"),
            (r"(设计|界面|用户体验|UI|UX)", ["ui-ux-designer"], "用户体验设计相关"),
            (r"(机器学习|AI|人工智能)", ["data-scientist"], "机器学习相关"),
            (r"(优化|改进|提升|增强)", ["adaptive3-thinking"], "优化改进相关")
        ]
        
        logger.info("🎯 智能体路由工作流初始化完成")

    async def execute_analysis(self, user_input: str) -> Dict[str, Any]:
        """执行智能体路由分析"""
        logger.info("🚀 开始智能体路由分析...")
        
        try:
            # 1. 意图识别和分析
            intent_analysis = await self._analyze_intent(user_input)
            
            # 2. 智能体推荐
            recommendations = await self._recommend_agents(intent_analysis, user_input)
            
            # 3. 生成路由结果
            routing_result = await self._generate_routing_result(
                user_input, intent_analysis, recommendations
            )
            
            # 4. 自动调用（如果启用）
            if self.config.auto_mode and recommendations:
                await self._auto_invoke_agent(recommendations[0], user_input)
            
            logger.info(f"✅ 智能体路由分析完成，推荐 {len(recommendations)} 个智能体")
            
            return routing_result
            
        except Exception as e:
            logger.error(f"❌ 智能体路由分析失败: {e}")
            return {
                "error": str(e),
                "user_input": user_input,
                "timestamp": datetime.now().isoformat()
            }

    async def _analyze_intent(self, user_input: str) -> IntentAnalysis:
        """意图分析"""
        logger.info("1️⃣ 意图识别和分析...")
        
        # 关键词匹配
        keywords_found = []
        confidence = 0.0
        
        for keyword, agent_info, score in self.agent_mapping.items():
            if keyword in user_input or keyword.lower() in user_input.lower():
                keywords_found.append({
                    "keyword": keyword,
                    "agent_id": agent_info[0],
                    "agent_name": agent_info[1]
                })
                confidence += score * 0.2
        
        # 复杂模式匹配
        complex_matches = []
        for pattern, agents, description in self.complex_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                complex_matches.append({
                    "pattern": pattern.pattern,
                    "agents": agents,
                    "description": description
                })
                confidence += 0.3
        
        # 实体识别
        entities = []
        if any(word in user_input for word in ["Python", "Java", "JavaScript", "C++"]):
            entities.append({"type": "programming_language", "value": "编程语言"})
        if any(word in user_input for word in ["数据库", "MySQL", "PostgreSQL"]):
            entities.append({"type": "database", "value": "数据库"})
        if any(word in user_input for word in ["Web", "移动", "桌面"]):
            entities.append({"type": "platform", "value": "平台"})
        
        # 确定主要意图
        primary_intent = "general"
        if keywords_found:
            primary_intent = keywords_found[0]["keyword"]
        elif complex_matches:
            primary_intent = complex_matches[0]["description"]
        
        confidence = min(confidence, 1.0)
        
        intent_result = IntentAnalysis(
            primary_intent=primary_intent,
            confidence=confidence,
            keywords=[k["keyword"] for k in keywords_found],
            entities=entities,
            explanation=f"识别到{len(keywords_found)}个关键词和{len(complex_matches)}个复杂模式"
        )
        
        logger.info(f"   ✅ 意图分析完成: {primary_intent} (置信度: {confidence:.2f})")
        
        return intent_result

    async def _recommend_agents(self, intent_analysis: IntentAnalysis, user_input: str) -> List[AgentRecommendation]:
        """智能体推荐"""
        logger.info("2️⃣ 智能体推荐...")
        
        recommendations = []
        
        # 基于主要意图推荐
        primary_intent = intent_analysis.primary_intent
        confidence = intent_analysis.confidence
        
        if primary_intent in self.agent_mapping:
            agent_info = self.agent_mapping[primary_intent]
            recommendations.append(AgentRecommendation(
                agent_id=agent_info[0],
                name=agent_info[1],
                confidence=confidence * agent_info[2],
                reason=f"匹配到{primary_intent}意图",
                parameters={"user_input": user_input}
            ))
        
        # 基于关键词推荐
        for keyword in intent_analysis.keywords:
            if keyword in self.agent_mapping and len(recommendations) < self.config.max_recommendations:
                agent_info = self.agent_mapping[keyword]
                if not any(r.agent_id == agent_info[0] for r in recommendations):
                    recommendations.append(AgentRecommendation(
                        agent_id=agent_info[0],
                        name=agent_info[1],
                        confidence=confidence * agent_info[2] * 0.8,
                        reason=f"关键词匹配: {keyword}",
                        parameters={"user_input": user_input}
                    ))
        
        # 基于复杂模式推荐
        for pattern, agents, description in self.complex_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                for agent_id in agents:
                    if len(recommendations) < self.config.max_recommendations:
                        if not any(r.agent_id == agent_id for r in recommendations):
                            recommendations.append(AgentRecommendation(
                                agent_id=agent_id,
                                name=f"{agent_id.replace('-', ' ').title()}",
                                confidence=confidence * 0.7,
                                reason=f"复杂模式匹配: {description}",
                                parameters={"user_input": user_input}
                            ))
        
        # 添加通用推荐
        if not recommendations:
            recommendations.append(AgentRecommendation(
                agent_id="chinese-commands",
                name="中文指令系统",
                confidence=0.5,
                reason="通用智能体",
                parameters={"user_input": user_input}
            ))
        
        # 按置信度排序
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        # 过滤低置信度推荐
        recommendations = [r for r in recommendations if r.confidence >= self.config.confidence_threshold]
        
        logger.info(f"   ✅ 推荐 {len(recommendations)} 个智能体")
        
        return recommendations

    async def _generate_routing_result(self, user_input: str, intent_analysis: IntentAnalysis, recommendations: List[AgentRecommendation]) -> Dict[str, Any]:
        """生成路由结果"""
        logger.info("3️⃣ 生成路由结果...")
        
        result = {
            "user_input": user_input,
            "intent_analysis": asdict(intent_analysis),
            "recommendations": [asdict(rec) for rec in recommendations],
            "selected_agent": asdict(recommendations[0]) if recommendations else None,
            "confidence": intent_analysis.confidence,
            "explanation": intent_analysis.explanation,
            "timestamp": datetime.now().isoformat(),
            "debug_info": {
                "config": asdict(self.config),
                "total_recommendations": len(recommendations),
                "confidence_threshold": self.config.confidence_threshold
            } if self.config.debug_mode else None
        }
        
        logger.info(f"   ✅ 路由结果生成完成")
        
        return result

    async def _auto_invoke_agent(self, recommendation: AgentRecommendation, user_input: str):
        """自动调用智能体"""
        logger.info(f"4️⃣ 自动调用智能体: {recommendation.name}")
        
        try:
            # 这里可以添加实际的智能体调用逻辑
            # 例如调用对应的MCP服务器或工作流
            
            auto_invoke_result = {
                "agent_id": recommendation.agent_id,
                "agent_name": recommendation.name,
                "status": "invoked",
                "parameters": recommendation.parameters,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"   ✅ 智能体自动调用成功: {recommendation.name}")
            
            return auto_invoke_result
            
        except Exception as e:
            logger.error(f"   ❌ 智能体自动调用失败: {e}")
            return {"error": str(e)}

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="智能体自动识别和调用系统工作流")
    parser.add_argument("--auto", action="store_true", help="自动模式")
    parser.add_argument("--manual", action="store_true", help="手动模式")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--input", required=True, help="用户输入")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="置信度阈值")
    parser.add_argument("--max-recommendations", type=int, default=3, help="最大推荐数量")
    
    args = parser.parse_args()
    
    # 创建路由配置
    config = RoutingConfig(
        auto_mode=args.auto,
        manual_mode=args.manual,
        debug_mode=args.debug,
        confidence_threshold=args.confidence_threshold,
        max_recommendations=args.max_recommendations
    )
    
    # 创建并执行路由工作流
    router = IntelligentRouterWorkflow(config)
    
    try:
        result = asyncio.run(router.execute_analysis(args.input))
        
        # 输出结果
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        
        return 0
        
    except Exception as e:
        logger.error(f"路由工作流执行失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)