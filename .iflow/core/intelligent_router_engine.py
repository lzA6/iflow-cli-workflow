#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能路由引擎 - 全自动智能体调用核心系统
基于用户意图自动识别和调用最适合的智能体
"""

import re
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import jieba
import jieba.analyse

class IntentType(Enum):
    """意图类型枚举"""
    ARCHITECTURE = "架构设计"
    PROGRAMMING = "编程开发"
    TESTING = "测试质量"
    SECURITY = "安全分析"
    DATA_ANALYSIS = "数据分析"
    THINKING = "思维决策"
    COLLABORATION = "协作沟通"
    TOOLS = "工具系统"
    BROWSING = "网络浏览"
    ANALYSIS = "分析评估"
    UNKNOWN = "未知意图"

@dataclass
class UserIntent:
    """用户意图数据类"""
    original_text: str
    intent_type: IntentType
    keywords: List[str]
    complexity: int  # 1-5级
    confidence: float  # 0-1置信度
    context: Dict[str, Any]
    timestamp: float

@dataclass
class AgentRecommendation:
    """智能体推荐数据类"""
    agent_name: str
    agent_type: str
    confidence: float
    reason: str
    priority: int  # 1-10优先级
    estimated_time: int  # 预估执行时间（秒）

class IntelligentRouter:
    """智能路由器核心类"""
    
    def __init__(self):
        self.intent_keywords = self._init_intent_keywords()
        self.agent_mapping = self._init_agent_mapping()
        self.user_history = {}
        self.learning_data = {}
        
    def _init_intent_keywords(self) -> Dict[IntentType, List[str]]:
        """初始化意图关键词映射"""
        return {
            IntentType.ARCHITECTURE: [
                "架构", "设计", "系统", "架构设计", "系统设计", "技术架构", 
                "微服务", "分布式", "企业架构", "IT架构", "技术选型", "技术栈"
            ],
            IntentType.PROGRAMMING: [
                "编程", "代码", "开发", "程序", "软件", "应用", "Python", 
                "JavaScript", "Java", "C++", "前端", "后端", "全栈", "算法"
            ],
            IntentType.TESTING: [
                "测试", "质量", "QA", "功能测试", "性能测试", "自动化测试",
                "代码覆盖率", "测试覆盖率", "缺陷", "bug", "问题", "调试"
            ],
            IntentType.SECURITY: [
                "安全", "漏洞", "风险", "安全审计", "安全检查", "渗透测试",
                "数据安全", "隐私", "加密", "认证", "授权", "防火墙"
            ],
            IntentType.DATA_ANALYSIS: [
                "数据", "分析", "统计", "机器学习", "AI", "模型", "数据挖掘",
                "预测", "可视化", "图表", "大数据", "数据科学", "算法"
            ],
            IntentType.THINKING: [
                "思考", "分析", "决策", "多维思考", "深度思考", "创新", "创意",
                "想法", "问题解决", "方案", "策略", "规划", "优化"
            ],
            IntentType.COLLABORATION: [
                "协作", "合作", "团队", "会议", "讨论", "沟通", "对话", "聊天",
                "交流", "文档", "报告", "总结", "分享", "演示"
            ],
            IntentType.TOOLS: [
                "命令", "CLI", "终端", "自动化", "脚本", "系统管理", "运维",
                "工具", "软件", "应用", "部署", "配置", "监控", "日志"
            ],
            IntentType.BROWSING: [
                "浏览器", "网页", "网站", "邮件", "邮箱", "邮件管理", "日历",
                "日程", "时间管理", "网页抓取", "数据采集", "网络", "在线"
            ],
            IntentType.ANALYSIS: [
                "分析", "评估", "评审", "进化", "改进", "优化", "反馈", "建议",
                "性能", "效率", "改进", "提升", "增强", "完善", "升级"
            ]
        }
    
    def _init_agent_mapping(self) -> Dict[IntentType, List[str]]:
        """初始化意图到智能体的映射"""
        return {
            IntentType.ARCHITECTURE: [
                "system-architect", "it-architect", "tech-stack-analyst"
            ],
            IntentType.PROGRAMMING: [
                "ai-programming-assistant", "fullstack-mentor", "code-coverage-analyst"
            ],
            IntentType.TESTING: [
                "quality-test-engineer", "code-coverage-analyst"
            ],
            IntentType.SECURITY: [
                "security-auditor"
            ],
            IntentType.DATA_ANALYSIS: [
                "data-scientist"
            ],
            IntentType.THINKING: [
                "adaptive3-thinking"
            ],
            IntentType.COLLABORATION: [
                "collaboration-mechanism", "live-meeting-co-pilot-cluely", "cluely-assistant"
            ],
            IntentType.TOOLS: [
                "interactive-cli-tool", "tool-master"
            ],
            IntentType.BROWSING: [
                "comet-browser-assistant"
            ],
            IntentType.ANALYSIS: [
                "arq-analyzer", "evolution-analyst", "mcp-feedback-enhanced"
            ]
        }
    
    def analyze_intent(self, user_input: str, context: Optional[Dict] = None) -> UserIntent:
        """分析用户意图"""
        # 文本预处理
        cleaned_text = self._preprocess_text(user_input)
        
        # 关键词提取
        keywords = jieba.analyse.extract_tags(cleaned_text, topK=10)
        
        # 意图识别
        intent_type = self._identify_intent(cleaned_text, keywords)
        
        # 复杂度评估
        complexity = self._assess_complexity(cleaned_text, keywords)
        
        # 置信度计算
        confidence = self._calculate_confidence(intent_type, keywords, cleaned_text)
        
        # 创建意图对象
        intent = UserIntent(
            original_text=user_input,
            intent_type=intent_type,
            keywords=keywords,
            complexity=complexity,
            confidence=confidence,
            context=context or {},
            timestamp=time.time()
        )
        
        return intent
    
    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 去除多余空格和特殊字符
        text = re.sub(r'\s+', ' ', text.strip())
        # 转换为小写
        text = text.lower()
        return text
    
    def _identify_intent(self, text: str, keywords: List[str]) -> IntentType:
        """识别意图类型"""
        intent_scores = {}
        
        for intent_type, intent_keywords in self.intent_keywords.items():
            score = 0
            for keyword in intent_keywords:
                if keyword in text:
                    score += 1
                for kw in keywords:
                    if keyword in kw or kw in keyword:
                        score += 0.5
            intent_scores[intent_type] = score
        
        # 找到得分最高的意图
        if not intent_scores or max(intent_scores.values()) == 0:
            return IntentType.UNKNOWN
        
        best_intent = max(intent_scores, key=intent_scores.get)
        return best_intent
    
    def _assess_complexity(self, text: str, keywords: List[str]) -> int:
        """评估任务复杂度"""
        complexity_score = 1
        
        # 基于文本长度
        if len(text) > 100:
            complexity_score += 1
        if len(text) > 200:
            complexity_score += 1
        
        # 基于关键词数量
        if len(keywords) > 5:
            complexity_score += 1
        if len(keywords) > 10:
            complexity_score += 1
        
        # 基于复杂词汇
        complex_keywords = ["系统", "架构", "复杂", "综合", "完整", "全面"]
        for keyword in complex_keywords:
            if keyword in text:
                complexity_score += 1
        
        return min(complexity_score, 5)
    
    def _calculate_confidence(self, intent_type: IntentType, keywords: List[str], text: str) -> float:
        """计算置信度"""
        if intent_type == IntentType.UNKNOWN:
            return 0.1
        
        # 基于关键词匹配度
        intent_keywords = self.intent_keywords[intent_type]
        match_count = sum(1 for kw in keywords if any(intent_kw in kw or kw in intent_kw for intent_kw in intent_keywords))
        
        if not keywords:
            return 0.1
        
        match_ratio = match_count / len(keywords)
        confidence = min(match_ratio + 0.2, 1.0)
        
        return confidence
    
    def recommend_agents(self, intent: UserIntent) -> List[AgentRecommendation]:
        """推荐智能体"""
        recommendations = []
        
        if intent.intent_type == IntentType.UNKNOWN:
            # 未知意图时推荐通用智能体
            recommendations.append(AgentRecommendation(
                agent_name="cluely-assistant",
                agent_type="通用助手",
                confidence=0.5,
                reason="意图不明确，推荐通用助手进行引导",
                priority=1,
                estimated_time=30
            ))
            return recommendations
        
        # 获取对应的智能体列表
        agent_names = self.agent_mapping.get(intent.intent_type, [])
        
        for agent_name in agent_names:
            confidence = self._calculate_agent_confidence(agent_name, intent)
            reason = self._generate_recommendation_reason(agent_name, intent)
            priority = self._calculate_priority(agent_name, intent)
            estimated_time = self._estimate_execution_time(agent_name, intent)
            
            recommendations.append(AgentRecommendation(
                agent_name=agent_name,
                agent_type=self._get_agent_type(agent_name),
                confidence=confidence,
                reason=reason,
                priority=priority,
                estimated_time=estimated_time
            ))
        
        # 按优先级排序
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        return recommendations
    
    def _calculate_agent_confidence(self, agent_name: str, intent: UserIntent) -> float:
        """计算智能体推荐置信度"""
        base_confidence = intent.confidence
        
        # 基于历史数据调整
        user_id = intent.context.get("user_id", "default")
        if user_id in self.user_history:
            agent_history = self.user_history[user_id].get(agent_name, {})
            success_rate = agent_history.get("success_rate", 0.8)
            base_confidence *= success_rate
        
        return min(base_confidence, 1.0)
    
    def _generate_recommendation_reason(self, agent_name: str, intent: UserIntent) -> str:
        """生成推荐理由"""
        reasons = {
            "system-architect": "擅长系统架构设计和技术选型",
            "ai-programming-assistant": "专业的编程助手，支持多种编程语言",
            "quality-test-engineer": "专注于软件质量保证和测试",
            "security-auditor": "专业的安全审计和风险评估",
            "data-scientist": "数据分析和机器学习专家",
            "adaptive3-thinking": "多维思考和决策分析专家",
            "fullstack-mentor": "全栈开发指导和教学",
            "cluely-assistant": "智能对话和任务协助助手",
            "live-meeting-co-pilot-cluely": "专业的会议管理和协作助手",
            "interactive-cli-tool": "命令行操作和自动化工具",
            "comet-browser-assistant": "浏览器自动化和网页管理助手",
            "tool-master": "系统工具管理和优化专家",
            "tech-stack-analyst": "技术栈分析和选型专家",
            "it-architect": "企业级IT架构设计专家",
            "code-coverage-analyst": "代码覆盖率分析专家",
            "collaboration-mechanism": "智能体协作机制专家",
            "mcp-feedback-enhanced": "反馈增强和性能优化专家",
            "arq-analyzer": "ARQ分析和推理专家",
            "evolution-analyst": "进化和改进分析专家"
        }
        
        return reasons.get(agent_name, "专业智能体，能够有效处理您的需求")
    
    def _calculate_priority(self, agent_name: str, intent: UserIntent) -> int:
        """计算优先级"""
        base_priority = 5
        
        # 基于复杂度调整
        if intent.complexity >= 4:
            base_priority += 2
        
        # 基于智能体专业度调整
        specialist_agents = {
            "system-architect": 2,
            "security-auditor": 2,
            "data-scientist": 2,
            "quality-test-engineer": 1
        }
        
        base_priority += specialist_agents.get(agent_name, 0)
        
        return min(base_priority, 10)
    
    def _estimate_execution_time(self, agent_name: str, intent: UserIntent) -> int:
        """估算执行时间"""
        base_time = 60  # 基础时间60秒
        
        # 基于复杂度调整
        complexity_multiplier = intent.complexity / 3
        
        # 基于智能体类型调整
        agent_multipliers = {
            "system-architect": 1.5,
            "data-scientist": 1.3,
            "security-auditor": 1.2,
            "quality-test-engineer": 1.1,
            "cluely-assistant": 0.8,
            "interactive-cli-tool": 0.7
        }
        
        multiplier = agent_multipliers.get(agent_name, 1.0)
        estimated_time = int(base_time * complexity_multiplier * multiplier)
        
        return estimated_time
    
    def _get_agent_type(self, agent_name: str) -> str:
        """获取智能体类型"""
        type_mapping = {
            "system-architect": "架构设计",
            "ai-programming-assistant": "编程开发",
            "quality-test-engineer": "测试质量",
            "security-auditor": "安全分析",
            "data-scientist": "数据分析",
            "adaptive3-thinking": "思维决策",
            "fullstack-mentor": "编程开发",
            "cluely-assistant": "通用助手",
            "live-meeting-co-pilot-cluely": "协作沟通",
            "interactive-cli-tool": "工具系统",
            "comet-browser-assistant": "网络浏览",
            "tool-master": "工具系统",
            "tech-stack-analyst": "架构设计",
            "it-architect": "架构设计",
            "code-coverage-analyst": "测试质量",
            "collaboration-mechanism": "协作沟通",
            "mcp-feedback-enhanced": "分析评估",
            "arq-analyzer": "分析评估",
            "evolution-analyst": "分析评估"
        }
        
        return type_mapping.get(agent_name, "通用类型")
    
    def route_and_execute(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """路由并执行任务"""
        # 分析意图
        intent = self.analyze_intent(user_input, context)
        
        # 推荐智能体
        recommendations = self.recommend_agents(intent)
        
        # 选择最佳智能体
        best_agent = recommendations[0] if recommendations else None
        
        if not best_agent:
            return {
                "status": "failed",
                "message": "无法找到合适的智能体来处理您的需求",
                "intent": intent,
                "suggestions": ["请更详细地描述您的需求", "尝试使用不同的关键词"]
            }
        
        # 构建响应
        result = {
            "status": "success",
            "intent": intent,
            "recommended_agent": best_agent,
            "all_recommendations": recommendations,
            "execution_plan": self._generate_execution_plan(intent, recommendations)
        }
        
        return result
    
    def _generate_execution_plan(self, intent: UserIntent, recommendations: List[AgentRecommendation]) -> Dict[str, Any]:
        """生成执行计划"""
        plan = {
            "primary_agent": recommendations[0].agent_name if recommendations else None,
            "complexity": intent.complexity,
            "estimated_time": sum(r.estimated_time for r in recommendations),
            "requires_collaboration": len(recommendations) > 1,
            "collaboration_agents": [r.agent_name for r in recommendations[1:]] if len(recommendations) > 1 else [],
            "steps": []
        }
        
        # 生成执行步骤
        if intent.complexity <= 2:
            plan["steps"] = [
                f"调用 {recommendations[0].agent_name} 处理您的问题",
                "返回处理结果"
            ]
        else:
            plan["steps"] = [
                f"1. 调用 {recommendations[0].agent_name} 进行主要分析",
                f"2. 根据需要调用其他智能体协作",
                "3. 整合所有结果",
                "4. 提供最终解决方案"
            ]
        
        return plan
    
    def learn_from_feedback(self, user_id: str, agent_name: str, feedback: Dict[str, Any]):
        """从反馈中学习"""
        if user_id not in self.user_history:
            self.user_history[user_id] = {}
        
        if agent_name not in self.user_history[user_id]:
            self.user_history[user_id][agent_name] = {
                "usage_count": 0,
                "success_rate": 0.8,
                "satisfaction_score": 0.8,
                "last_used": 0
            }
        
        # 更新历史数据
        agent_history = self.user_history[user_id][agent_name]
        agent_history["usage_count"] += 1
        agent_history["last_used"] = time.time()
        
        # 更新成功率和满意度
        if "success" in feedback:
            current_success_rate = agent_history["success_rate"]
            agent_history["success_rate"] = (current_success_rate * 0.9 + feedback["success"] * 0.1)
        
        if "satisfaction" in feedback:
            current_satisfaction = agent_history["satisfaction_score"]
            agent_history["satisfaction_score"] = (current_satisfaction * 0.9 + feedback["satisfaction"] * 0.1)

# 智能路由器实例
router = IntelligentRouter()

def route_user_request(user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """路由用户请求的便捷函数"""
    return router.route_and_execute(user_input, context)

def get_agent_recommendations(user_input: str) -> List[AgentRecommendation]:
    """获取智能体推荐的便捷函数"""
    intent = router.analyze_intent(user_input)
    return router.recommend_agents(intent)

def submit_feedback(user_id: str, agent_name: str, feedback: Dict[str, Any]):
    """提交反馈的便捷函数"""
    router.learn_from_feedback(user_id, agent_name, feedback)

if __name__ == "__main__":
    # 测试示例
    test_inputs = [
        "我想设计一个电商系统",
        "帮我写一个Python程序",
        "我的代码有bug，帮我调试",
        "我想分析一下销售数据",
        "我们需要开个会讨论项目"
    ]
    
    for test_input in test_inputs:
        print(f"用户输入: {test_input}")
        result = route_user_request(test_input)
        print(f"推荐智能体: {result['recommended_agent'].agent_name}")
        print(f"推荐理由: {result['recommended_agent'].reason}")
        print("-" * 50)