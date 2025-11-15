#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 Meta-Agent治理层 V11 (代号："守护者")
===========================================================

这是 T-MIA 架构下的最高治理层，负责定义和管理智能体之间的协作规则、权限和优先级。
V11版本实现了真正的系统级自我治理、动态规则演化和多维度治理机制。

核心特性：
- 系统级自我治理 - 定义和修改智能体协作规则
- 动态规则演化 - 根据系统表现自动调整治理规则
- 多维度治理 - 从性能、安全、质量等多个维度治理
- 权限管理 - 精细化的智能体权限控制
- 优先级调度 - 智能的任务优先级管理

你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。

作者: AI架构师团队
版本: 11.0.0 (代号："守护者")
日期: 2025-11-15
"""

import os
import sys
import json
import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

# 项目根路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetaAgentGovernorV11")

class GovernanceDomain(Enum):
    """治理域"""
    PERFORMANCE = "performance"
    SECURITY = "security"
    QUALITY = "quality"
    COLLABORATION = "collaboration"
    RESOURCE = "resource"
    EVOLUTION = "evolution"

class AgentRole(Enum):
    """智能体角色"""
    WORKER = "worker"
    COORDINATOR = "coordinator"
    SUPERVISOR = "supervisor"
    GOVERNOR = "governor"
    ORACLE = "oracle"

class PermissionLevel(Enum):
    """权限级别"""
    NONE = 0
    READ = 1
    WRITE = 2
    EXECUTE = 3
    ADMIN = 4
    SUPERADMIN = 5

@dataclass
class GovernanceRule:
    """治理规则"""
    rule_id: str
    domain: GovernanceDomain
    name: str
    description: str
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    priority: int
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    effectiveness_score: float = 0.0
    application_count: int = 0

@dataclass
class AgentProfile:
    """智能体档案"""
    agent_id: str
    name: str
    role: AgentRole
    capabilities: List[str]
    permissions: Dict[GovernanceDomain, PermissionLevel]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    collaboration_history: List[Dict[str, Any]] = field(default_factory=list)
    trust_score: float = 0.5
    reliability_score: float = 0.5
    last_active: Optional[datetime] = None

@dataclass
class GovernanceDecision:
    """治理决策"""
    decision_id: str
    timestamp: datetime
    domain: GovernanceDomain
    context: Dict[str, Any]
    rules_applied: List[str]
    decision: str
    rationale: str
    impact_assessment: Dict[str, Any]
    feedback_score: Optional[float] = None

class MetaAgentGovernorV11:
    """Meta-Agent治理层 V11"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 治理规则
        self.governance_rules: Dict[str, GovernanceRule] = {}
        self.rule_evolution_history = deque(maxlen=1000)
        
        # 智能体管理
        self.agent_profiles: Dict[str, AgentProfile] = {}
        self.agent_hierarchy = nx.DiGraph()
        self.collaboration_graph = nx.Graph()
        
        # 决策记录
        self.governance_decisions: Dict[str, GovernanceDecision] = {}
        self.decision_patterns = defaultdict(list)
        
        # 治理指标
        self.governance_metrics = defaultdict(float)
        self.domain_health_scores = defaultdict(float)
        
        # 自适应机制
        self.learning_rate = 0.01
        self.adaptation_threshold = 0.7
        self.evolution_cycle = 3600  # 1小时
        
        # 权限矩阵
        self.permission_matrix = defaultdict(lambda: PermissionLevel.NONE)
        
        logger.info("Meta-Agent治理层V11初始化完成")
    
    async def initialize(self):
        """异步初始化"""
        logger.info("正在初始化Meta-Agent治理层...")
        
        # 加载基础治理规则
        await self._load_base_governance_rules()
        
        # 初始化智能体档案
        await self._initialize_agent_profiles()
        
        # 构建智能体层次结构
        await self._build_agent_hierarchy()
        
        # 启动治理循环
        asyncio.create_task(self._governance_loop())
        asyncio.create_task(self._rule_evolution_loop())
        asyncio.create_task(self._agent_monitoring_loop())
        asyncio.create_task(self._decision_analysis_loop())
        
        logger.info("Meta-Agent治理层初始化完成")
    
    async def register_agent(self, 
                           agent_id: str,
                           name: str,
                           role: AgentRole,
                           capabilities: List[str],
                           initial_permissions: Optional[Dict[GovernanceDomain, PermissionLevel]] = None) -> bool:
        """注册智能体"""
        try:
            # 创建智能体档案
            profile = AgentProfile(
                agent_id=agent_id,
                name=name,
                role=role,
                capabilities=capabilities,
                permissions=initial_permissions or defaultdict(lambda: PermissionLevel.READ),
                last_active=datetime.now()
            )
            
            self.agent_profiles[agent_id] = profile
            
            # 添加到层次结构
            self.agent_hierarchy.add_node(agent_id, profile=profile)
            
            # 设置默认权限
            await self._set_default_permissions(agent_id, role)
            
            # 添加到协作图
            self.collaboration_graph.add_node(agent_id)
            
            logger.info(f"注册智能体成功: {agent_id} ({name})")
            return True
            
        except Exception as e:
            logger.error(f"注册智能体失败 {agent_id}: {e}")
            return False
    
    async def govern_agent_action(self, 
                                agent_id: str,
                                action: str,
                                domain: GovernanceDomain,
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """治理智能体行动"""
        governance_result = {
            'permitted': False,
            'rules_applied': [],
            'conditions': [],
            'modifications': [],
            'decision_id': None
        }
        
        try:
            # 检查智能体是否存在
            if agent_id not in self.agent_profiles:
                governance_result['reason'] = f"智能体未注册: {agent_id}"
                return governance_result
            
            profile = self.agent_profiles[agent_id]
            
            # 检查权限
            if not await self._check_permission(agent_id, action, domain):
                governance_result['reason'] = f"权限不足: {action} in {domain.value}"
                return governance_result
            
            # 应用治理规则
            applicable_rules = await self._get_applicable_rules(agent_id, action, domain, context)
            
            decision_made = False
            for rule in applicable_rules:
                if await self._evaluate_rule_condition(rule, context):
                    # 应用规则
                    rule_result = await self._apply_governance_rule(rule, agent_id, action, context)
                    
                    governance_result['rules_applied'].append(rule.rule_id)
                    governance_result['conditions'].extend(rule_result.get('conditions', []))
                    governance_result['modifications'].extend(rule_result.get('modifications', []))
                    
                    # 更新规则统计
                    rule.application_count += 1
                    rule.last_modified = datetime.now()
                    
                    if not decision_made:
                        # 记录治理决策
                        decision = await self._make_governance_decision(
                            agent_id, action, domain, context, rule
                        )
                        governance_result['decision_id'] = decision.decision_id
                        governance_result['permitted'] = decision.decision == 'permit'
                        decision_made = True
            
            # 如果没有规则适用，使用默认决策
            if not decision_made:
                decision = await self._make_default_decision(agent_id, action, domain, context)
                governance_result['decision_id'] = decision.decision_id
                governance_result['permitted'] = decision.decision == 'permit'
            
            # 更新智能体活动记录
            profile.last_active = datetime.now()
            await self._update_agent_metrics(agent_id, governance_result)
            
            return governance_result
            
        except Exception as e:
            logger.error(f"治理智能体行动失败 {agent_id}: {e}")
            governance_result['reason'] = f"治理过程异常: {str(e)}"
            return governance_result
    
    async def evolve_governance_rules(self):
        """演化治理规则"""
        logger.info("开始演化治理规则...")
        
        # 分析规则效果
        rule_effectiveness = await self._analyze_rule_effectiveness()
        
        # 识别需要改进的规则
        rules_to_improve = [
            rule_id for rule_id, effectiveness in rule_effectiveness.items()
            if effectiveness < self.adaptation_threshold
        ]
        
        # 生成新规则
        for rule_id in rules_to_improve:
            old_rule = self.governance_rules[rule_id]
            new_rule = await self._generate_improved_rule(old_rule, rule_effectiveness[rule_id])
            
            if new_rule:
                # 禁用旧规则
                old_rule.enabled = False
                
                # 启用新规则
                self.governance_rules[new_rule.rule_id] = new_rule
                
                # 记录演化历史
                self.rule_evolution_history.append({
                    'timestamp': datetime.now(),
                    'old_rule_id': rule_id,
                    'new_rule_id': new_rule.rule_id,
                    'reason': 'performance_improvement',
                    'old_effectiveness': rule_effectiveness[rule_id]
                })
                
                logger.info(f"演化规则: {rule_id} -> {new_rule.rule_id}")
        
        # 探索全新的规则
        if random.random() < 0.1:  # 10%概率探索新规则
            new_rule = await self._explore_new_rule()
            if new_rule:
                self.governance_rules[new_rule.rule_id] = new_rule
                logger.info(f"探索新规则: {new_rule.rule_id}")
        
        logger.info(f"规则演化完成，当前规则数: {len(self.governance_rules)}")
    
    async def _load_base_governance_rules(self):
        """加载基础治理规则"""
        # 性能域规则
        self.governance_rules['perf_001'] = GovernanceRule(
            rule_id='perf_001',
            domain=GovernanceDomain.PERFORMANCE,
            name='任务执行时间限制',
            description='限制单个任务的执行时间以防止系统阻塞',
            conditions={
                'task_type': 'computation_heavy',
                'estimated_duration': {'>': 300}  # 5分钟
            },
            actions=[
                {'action': 'require_optimization', 'level': 'moderate'},
                {'action': 'enable_monitoring', 'interval': 30}
            ],
            priority=1
        )
        
        # 安全域规则
        self.governance_rules['sec_001'] = GovernanceRule(
            rule_id='sec_001',
            domain=GovernanceDomain.SECURITY,
            name='敏感操作验证',
            description='敏感操作需要多重验证',
            conditions={
                'operation_type': 'sensitive',
                'data_classification': {'>=': 'confidential'}
            },
            actions=[
                {'action': 'require_multi_factor_auth'},
                {'action': 'log_detailed_audit'}
            ],
            priority=2
        )
        
        # 质量域规则
        self.governance_rules['qual_001'] = GovernanceRule(
            rule_id='qual_001',
            domain=GovernanceDomain.QUALITY,
            name='代码质量检查',
            description='重要代码变更需要质量检查',
            conditions={
                'change_type': 'code_modification',
                'impact_level': {'>=': 'medium'}
            },
            actions=[
                {'action': 'require_code_review'},
                {'action': 'run_quality_tests'}
            ],
            priority=1
        )
        
        # 协作域规则
        self.governance_rules['col_001'] = GovernanceRule(
            rule_id='col_001',
            domain=GovernanceDomain.COLLABORATION,
            name='协作冲突解决',
            description='智能体间的协作冲突需要自动解决',
            conditions={
                'conflict_type': 'resource_competition',
                'involved_agents': {'>=': 2}
            },
            actions=[
                {'action': 'apply_priority_resolution'},
                {'action': 'enable_negotiation'}
            ],
            priority=1
        )
        
        logger.info(f"加载了 {len(self.governance_rules)} 个基础治理规则")
    
    async def _initialize_agent_profiles(self):
        """初始化智能体档案"""
        # 注册核心智能体
        await self.register_agent(
            agent_id='arq_analyzer',
            name='ARQ分析器',
            role=AgentRole.WORKER,
            capabilities=['arq_analysis', 'reasoning', 'pattern_recognition']
        )
        
        await self.register_agent(
            agent_id='workflow_engine',
            name='工作流引擎',
            role=AgentRole.COORDINATOR,
            capabilities=['workflow_orchestration', 'task_scheduling', 'adaptation']
        )
        
        await self.register_agent(
            agent_id='consciousness_system',
            name='意识流系统',
            role=AgentRole.SUPERVISOR,
            capabilities=['context_management', 'memory_compression', 'emotional_reasoning']
        )
        
        await self.register_agent(
            agent_id='meta_governor',
            name='元治理者',
            role=AgentRole.GOVERNOR,
            capabilities=['rule_evolution', 'agent_management', 'system_optimization']
        )
        
        logger.info(f"初始化了 {len(self.agent_profiles)} 个智能体档案")
    
    async def _build_agent_hierarchy(self):
        """构建智能体层次结构"""
        # 建立层次关系
        hierarchy_relations = [
            ('meta_governor', 'consciousness_system'),
            ('consciousness_system', 'workflow_engine'),
            ('workflow_engine', 'arq_analyzer')
        ]
        
        for supervisor, subordinate in hierarchy_relations:
            if supervisor in self.agent_profiles and subordinate in self.agent_profiles:
                self.agent_hierarchy.add_edge(supervisor, subordinate)
        
        logger.info("智能体层次结构构建完成")
    
    async def _set_default_permissions(self, agent_id: str, role: AgentRole):
        """设置默认权限"""
        role_permissions = {
            AgentRole.WORKER: {
                GovernanceDomain.PERFORMANCE: PermissionLevel.READ,
                GovernanceDomain.SECURITY: PermissionLevel.READ,
                GovernanceDomain.QUALITY: PermissionLevel.READ,
                GovernanceDomain.COLLABORATION: PermissionLevel.WRITE,
                GovernanceDomain.RESOURCE: PermissionLevel.READ,
                GovernanceDomain.EVOLUTION: PermissionLevel.NONE
            },
            AgentRole.COORDINATOR: {
                GovernanceDomain.PERFORMANCE: PermissionLevel.WRITE,
                GovernanceDomain.SECURITY: PermissionLevel.READ,
                GovernanceDomain.QUALITY: PermissionLevel.WRITE,
                GovernanceDomain.COLLABORATION: PermissionLevel.EXECUTE,
                GovernanceDomain.RESOURCE: PermissionLevel.WRITE,
                GovernanceDomain.EVOLUTION: PermissionLevel.READ
            },
            AgentRole.SUPERVISOR: {
                GovernanceDomain.PERFORMANCE: PermissionLevel.EXECUTE,
                GovernanceDomain.SECURITY: PermissionLevel.WRITE,
                GovernanceDomain.QUALITY: PermissionLevel.EXECUTE,
                GovernanceDomain.COLLABORATION: PermissionLevel.EXECUTE,
                GovernanceDomain.RESOURCE: PermissionLevel.EXECUTE,
                GovernanceDomain.EVOLUTION: PermissionLevel.WRITE
            },
            AgentRole.GOVERNOR: {
                GovernanceDomain.PERFORMANCE: PermissionLevel.SUPERADMIN,
                GovernanceDomain.SECURITY: PermissionLevel.SUPERADMIN,
                GovernanceDomain.QUALITY: PermissionLevel.SUPERADMIN,
                GovernanceDomain.COLLABORATION: PermissionLevel.SUPERADMIN,
                GovernanceDomain.RESOURCE: PermissionLevel.SUPERADMIN,
                GovernanceDomain.EVOLUTION: PermissionLevel.SUPERADMIN
            }
        }
        
        if role in role_permissions:
            self.agent_profiles[agent_id].permissions = role_permissions[role]
    
    async def _check_permission(self, agent_id: str, action: str, domain: GovernanceDomain) -> bool:
        """检查权限"""
        if agent_id not in self.agent_profiles:
            return False
        
        profile = self.agent_profiles[agent_id]
        required_permission = self._get_required_permission(action)
        
        return profile.permissions.get(domain, PermissionLevel.NONE) >= required_permission
    
    def _get_required_permission(self, action: str) -> PermissionLevel:
        """获取所需权限级别"""
        action_permissions = {
            'read': PermissionLevel.READ,
            'write': PermissionLevel.WRITE,
            'execute': PermissionLevel.EXECUTE,
            'admin': PermissionLevel.ADMIN,
            'modify_rules': PermissionLevel.SUPERADMIN
        }
        
        return action_permissions.get(action, PermissionLevel.READ)
    
    async def _get_applicable_rules(self, 
                                  agent_id: str,
                                  action: str,
                                  domain: GovernanceDomain,
                                  context: Dict[str, Any]) -> List[GovernanceRule]:
        """获取适用的治理规则"""
        applicable_rules = []
        
        for rule in self.governance_rules.values():
            if not rule.enabled:
                continue
            
            # 检查域匹配
            if rule.domain != domain:
                continue
            
            # 检查智能体角色匹配
            if not await self._rule_applies_to_agent(rule, agent_id):
                continue
            
            # 检查条件匹配
            if await self._rule_condition_matches(rule, context):
                applicable_rules.append(rule)
        
        # 按优先级排序
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)
        
        return applicable_rules
    
    async def _rule_applies_to_agent(self, rule: GovernanceRule, agent_id: str) -> bool:
        """检查规则是否适用于智能体"""
        profile = self.agent_profiles.get(agent_id)
        if not profile:
            return False
        
        # 检查角色限制
        if 'agent_roles' in rule.conditions:
            required_roles = rule.conditions['agent_roles']
            if profile.role not in required_roles:
                return False
        
        # 检查能力要求
        if 'required_capabilities' in rule.conditions:
            required_caps = rule.conditions['required_capabilities']
            if not any(cap in profile.capabilities for cap in required_caps):
                return False
        
        return True
    
    async def _rule_condition_matches(self, rule: GovernanceRule, context: Dict[str, Any]) -> bool:
        """检查规则条件是否匹配"""
        conditions = rule.conditions
        
        for key, condition in conditions.items():
            if key not in context:
                continue
            
            context_value = context[key]
            
            if isinstance(condition, dict):
                # 处理比较操作
                for op, value in condition.items():
                    if op == '>' and not context_value > value:
                        return False
                    elif op == '<' and not context_value < value:
                        return False
                    elif op == '>=' and not context_value >= value:
                        return False
                    elif op == '<=' and not context_value <= value:
                        return False
                    elif op == '==' and not context_value == value:
                        return False
                    elif op == '!=' and not context_value != value:
                        return False
                    elif op == 'in' and context_value not in value:
                        return False
            elif context_value != condition:
                return False
        
        return True
    
    async def _evaluate_rule_condition(self, rule: GovernanceRule, context: Dict[str, Any]) -> bool:
        """评估规则条件"""
        return await self._rule_condition_matches(rule, context)
    
    async def _apply_governance_rule(self, 
                                   rule: GovernanceRule,
                                   agent_id: str,
                                   action: str,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """应用治理规则"""
        result = {
            'conditions': [],
            'modifications': []
        }
        
        for action_def in rule.actions:
            action_type = action_def.get('action')
            
            if action_type == 'require_optimization':
                result['conditions'].append('optimization_required')
                result['modifications'].append({'type': 'optimize', 'level': action_def.get('level', 'moderate')})
            
            elif action_type == 'enable_monitoring':
                result['conditions'].append('monitoring_enabled')
                result['modifications'].append({'type': 'monitor', 'interval': action_def.get('interval', 60)})
            
            elif action_type == 'require_multi_factor_auth':
                result['conditions'].append('mfa_required')
                result['modifications'].append({'type': 'auth', 'method': 'multi_factor'})
            
            elif action_type == 'log_detailed_audit':
                result['conditions'].append('audit_logging_enabled')
                result['modifications'].append({'type': 'logging', 'level': 'detailed'})
            
            elif action_type == 'require_code_review':
                result['conditions'].append('code_review_required')
                result['modifications'].append({'type': 'review', 'mandatory': True})
            
            elif action_type == 'run_quality_tests':
                result['conditions'].append('quality_tests_required')
                result['modifications'].append({'type': 'test', 'coverage': 'full'})
            
            elif action_type == 'apply_priority_resolution':
                result['conditions'].append('priority_resolution_enabled')
                result['modifications'].append({'type': 'resolution', 'method': 'priority_based'})
            
            elif action_type == 'enable_negotiation':
                result['conditions'].append('negotiation_enabled')
                result['modifications'].append({'type': 'negotiation', 'protocol': 'collaborative'})
        
        return result
    
    async def _make_governance_decision(self, 
                                      agent_id: str,
                                      action: str,
                                      domain: GovernanceDomain,
                                      context: Dict[str, Any],
                                      rule: GovernanceRule) -> GovernanceDecision:
        """制定治理决策"""
        decision_id = str(uuid.uuid4())
        
        # 基于规则和上下文制定决策
        decision = 'permit'  # 默认允许
        
        # 风险评估
        risk_score = await self._assess_action_risk(agent_id, action, domain, context)
        
        if risk_score > 0.8:
            decision = 'deny'
            rationale = '高风险操作被拒绝'
        elif risk_score > 0.6:
            decision = 'conditional'
            rationale = '需要额外条件才能执行'
        else:
            decision = 'permit'
            rationale = '操作符合治理规则'
        
        # 影响评估
        impact = await self._assess_decision_impact(decision, agent_id, action, domain)
        
        governance_decision = GovernanceDecision(
            decision_id=decision_id,
            timestamp=datetime.now(),
            domain=domain,
            context=context.copy(),
            rules_applied=[rule.rule_id],
            decision=decision,
            rationale=rationale,
            impact_assessment=impact
        )
        
        self.governance_decisions[decision_id] = governance_decision
        
        # 记录决策模式
        self.decision_patterns[f"{domain.value}_{action}"].append({
            'timestamp': datetime.now(),
            'decision': decision,
            'risk_score': risk_score,
            'agent_role': self.agent_profiles[agent_id].role.value
        })
        
        return governance_decision
    
    async def _make_default_decision(self, 
                                   agent_id: str,
                                   action: str,
                                   domain: GovernanceDomain,
                                   context: Dict[str, Any]) -> GovernanceDecision:
        """制定默认决策"""
        decision_id = str(uuid.uuid4())
        
        # 基于智能体角色和信任度制定默认决策
        profile = self.agent_profiles[agent_id]
        
        if profile.trust_score > 0.7 and profile.reliability_score > 0.7:
            decision = 'permit'
            rationale = '高信任度和可靠性的智能体'
        elif profile.trust_score > 0.5:
            decision = 'conditional'
            rationale = '中等信任度，需要监控'
        else:
            decision = 'deny'
            rationale = '信任度不足'
        
        governance_decision = GovernanceDecision(
            decision_id=decision_id,
            timestamp=datetime.now(),
            domain=domain,
            context=context.copy(),
            rules_applied=[],
            decision=decision,
            rationale=rationale,
            impact_assessment={'risk_level': 'unknown'}
        )
        
        self.governance_decisions[decision_id] = governance_decision
        
        return governance_decision
    
    async def _assess_action_risk(self, 
                                agent_id: str,
                                action: str,
                                domain: GovernanceDomain,
                                context: Dict[str, Any]) -> float:
        """评估行动风险"""
        base_risk = 0.3
        
        # 基于域的风险调整
        domain_risks = {
            GovernanceDomain.SECURITY: 0.5,
            GovernanceDomain.RESOURCE: 0.3,
            GovernanceDomain.PERFORMANCE: 0.2,
            GovernanceDomain.QUALITY: 0.2,
            GovernanceDomain.COLLABORATION: 0.1,
            GovernanceDomain.EVOLUTION: 0.4
        }
        
        base_risk += domain_risks.get(domain, 0.0)
        
        # 基于智能体特征调整
        profile = self.agent_profiles[agent_id]
        risk_adjustment = (1.0 - profile.trust_score) * 0.3
        risk_adjustment += (1.0 - profile.reliability_score) * 0.2
        
        # 基于历史表现调整
        recent_decisions = [
            d for d in self.governance_decisions.values()
            if d.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        if recent_decisions:
            failure_rate = sum(1 for d in recent_decisions if d.decision == 'deny') / len(recent_decisions)
            risk_adjustment += failure_rate * 0.2
        
        total_risk = min(1.0, base_risk + risk_adjustment)
        
        return total_risk
    
    async def _assess_decision_impact(self, 
                                    decision: str,
                                    agent_id: str,
                                    action: str,
                                    domain: GovernanceDomain) -> Dict[str, Any]:
        """评估决策影响"""
        impact = {
            'risk_level': 'low',
            'affected_agents': [agent_id],
            'system_impact': 'minimal',
            'duration': 'short_term'
        }
        
        if decision == 'deny':
            impact['risk_level'] = 'low'
            impact['system_impact'] = 'prevention'
        elif decision == 'conditional':
            impact['risk_level'] = 'medium'
            impact['system_impact'] = 'controlled'
        else:  # permit
            impact['risk_level'] = 'medium'
            impact['system_impact'] = 'operational'
        
        # 基于域调整影响
        if domain == GovernanceDomain.SECURITY:
            impact['system_impact'] = 'critical' if decision == 'permit' else 'protective'
        elif domain == GovernanceDomain.EVOLUTION:
            impact['duration'] = 'long_term'
        
        return impact
    
    async def _update_agent_metrics(self, agent_id: str, governance_result: Dict[str, Any]):
        """更新智能体指标"""
        if agent_id not in self.agent_profiles:
            return
        
        profile = self.agent_profiles[agent_id]
        
        # 更新信任度
        if governance_result.get('permitted', False):
            profile.trust_score = min(1.0, profile.trust_score + 0.01)
        else:
            profile.trust_score = max(0.0, profile.trust_score - 0.02)
        
        # 更新可靠性
        if not governance_result.get('reason'):  # 没有错误
            profile.reliability_score = min(1.0, profile.reliability_score + 0.01)
        
        # 记录协作历史
        profile.collaboration_history.append({
            'timestamp': datetime.now(),
            'action': governance_result,
            'outcome': 'success' if governance_result.get('permitted', False) else 'blocked'
        })
        
        # 限制历史记录大小
        if len(profile.collaboration_history) > 100:
            profile.collaboration_history = profile.collaboration_history[-100:]
    
    async def _analyze_rule_effectiveness(self) -> Dict[str, float]:
        """分析规则效果"""
        effectiveness = {}
        
        for rule_id, rule in self.governance_rules.items():
            if rule.application_count == 0:
                effectiveness[rule_id] = 0.5  # 中性评分
                continue
            
            # 基于反馈评分计算效果
            recent_decisions = [
                d for d in self.governance_decisions.values()
                if rule_id in d.rules_applied and 
                   d.timestamp > datetime.now() - timedelta(days=7)
            ]
            
            if recent_decisions:
                feedback_scores = [d.feedback_score or 0.5 for d in recent_decisions]
                avg_feedback = sum(feedback_scores) / len(feedback_scores)
                
                # 结合应用频率
                frequency_factor = min(1.0, rule.application_count / 10.0)
                
                effectiveness[rule_id] = avg_feedback * 0.7 + frequency_factor * 0.3
            else:
                effectiveness[rule_id] = rule.effectiveness_score
        
        return effectiveness
    
    async def _generate_improved_rule(self, old_rule: GovernanceRule, effectiveness: float) -> Optional[GovernanceRule]:
        """生成改进的规则"""
        if effectiveness > 0.5:
            return None  # 不需要改进
        
        # 分析失败原因
        failure_patterns = await self._analyze_rule_failures(old_rule)
        
        # 生成改进版本
        new_rule = GovernanceRule(
            rule_id=f"{old_rule.rule_id}_v2",
            domain=old_rule.domain,
            name=f"{old_rule.name} (改进版)",
            description=f"基于效果评估改进的规则，原效果: {effectiveness:.2f}",
            conditions=old_rule.conditions.copy(),
            actions=old_rule.actions.copy(),
            priority=old_rule.priority,
            effectiveness_score=0.6  # 期望效果
        )
        
        # 根据失败模式调整规则
        for pattern in failure_patterns:
            if pattern['type'] == 'too_strict':
                # 放宽条件
                await self._relax_rule_conditions(new_rule, pattern['details'])
            elif pattern['type'] == 'too_permissive':
                # 加强条件
                await self._tighten_rule_conditions(new_rule, pattern['details'])
            elif pattern['type'] == 'wrong_actions':
                # 调整行动
                await self._modify_rule_actions(new_rule, pattern['details'])
        
        return new_rule
    
    async def _analyze_rule_failures(self, rule: GovernanceRule) -> List[Dict[str, Any]]:
        """分析规则失败原因"""
        failures = []
        
        # 获取相关决策
        related_decisions = [
            d for d in self.governance_decisions.values()
            if rule.rule_id in d.rules_applied
        ]
        
        if not related_decisions:
            return failures
        
        # 分析反馈
        negative_feedback = [
            d for d in related_decisions
            if d.feedback_score and d.feedback_score < 0.5
        ]
        
        if len(negative_feedback) / len(related_decisions) > 0.6:
            if 'too strict' in [d.rationale.lower() for d in negative_feedback]:
                failures.append({'type': 'too_strict', 'details': '规则过于严格'})
            elif 'too permissive' in [d.rationale.lower() for d in negative_feedback]:
                failures.append({'type': 'too_permissive', 'details': '规则过于宽松'})
            else:
                failures.append({'type': 'wrong_actions', 'details': '行动不适当'})
        
        return failures
    
    async def _relax_rule_conditions(self, rule: GovernanceRule, details: str):
        """放宽规则条件"""
        # 示例：调整数值阈值
        for key, condition in rule.conditions.items():
            if isinstance(condition, dict) and '>' in condition:
                condition['>'] *= 0.8  # 降低阈值
    
    async def _tighten_rule_conditions(self, rule: GovernanceRule, details: str):
        """加强规则条件"""
        # 示例：调整数值阈值
        for key, condition in rule.conditions.items():
            if isinstance(condition, dict) and '>' in condition:
                condition['>'] *= 1.2  # 提高阈值
    
    async def _modify_rule_actions(self, rule: GovernanceRule, details: str):
        """修改规则行动"""
        # 示例：添加监控行动
        if not any('monitoring' in action.get('action', '') for action in rule.actions):
            rule.actions.append({
                'action': 'enable_monitoring',
                'interval': 60
            })
    
    async def _explore_new_rule(self) -> Optional[GovernanceRule]:
        """探索新规则"""
        # 基于决策模式生成新规则
        common_patterns = [
            pattern for pattern, decisions in self.decision_patterns.items()
            if len(decisions) > 5
        ]
        
        if not common_patterns:
            return None
        
        # 选择最频繁的模式
        pattern = max(common_patterns, key=lambda p: len(self.decision_patterns[p]))
        decisions = self.decision_patterns[pattern]
        
        # 分析模式特征
        domain, action = pattern.split('_', 1)
        avg_risk = sum(d['risk_score'] for d in decisions) / len(decisions)
        
        # 生成新规则
        if avg_risk > 0.7:
            new_rule = GovernanceRule(
                rule_id=f"auto_{int(time.time())}",
                domain=GovernanceDomain(domain),
                name=f"自动生成的{action}规则",
                description="基于决策模式自动生成",
                conditions={
                    'action': action,
                    'risk_threshold': {'>': avg_risk * 0.8}
                },
                actions=[
                    {'action': 'require_additional_validation'},
                    {'action': 'enable_enhanced_monitoring'}
                ],
                priority=1,
                effectiveness_score=0.5
            )
            
            return new_rule
        
        return None
    
    async def _governance_loop(self):
        """治理循环"""
        while True:
            try:
                await asyncio.sleep(300)  # 5分钟
                
                # 更新治理指标
                await self._update_governance_metrics()
                
                # 检查系统健康
                await self._check_system_health()
                
                # 处理待决策
                await self._process_pending_decisions()
                
            except Exception as e:
                logger.error(f"治理循环错误: {e}")
    
    async def _rule_evolution_loop(self):
        """规则演化循环"""
        while True:
            try:
                await asyncio.sleep(self.evolution_cycle)  # 1小时
                
                # 演化规则
                await self.evolve_governance_rules()
                
                # 清理过期规则
                await self._cleanup_expired_rules()
                
            except Exception as e:
                logger.error(f"规则演化循环错误: {e}")
    
    async def _agent_monitoring_loop(self):
        """智能体监控循环"""
        while True:
            try:
                await asyncio.sleep(60)  # 1分钟
                
                # 监控智能体活动
                await self._monitor_agent_activity()
                
                # 更新协作图
                await self._update_collaboration_graph()
                
                # 检测异常行为
                await self._detect_anomalous_behavior()
                
            except Exception as e:
                logger.error(f"智能体监控循环错误: {e}")
    
    async def _decision_analysis_loop(self):
        """决策分析循环"""
        while True:
            try:
                await asyncio.sleep(600)  # 10分钟
                
                # 分析决策模式
                await self._analyze_decision_patterns()
                
                # 收集反馈
                await self._collect_decision_feedback()
                
                # 优化决策策略
                await self._optimize_decision_strategies()
                
            except Exception as e:
                logger.error(f"决策分析循环错误: {e}")
    
    async def _update_governance_metrics(self):
        """更新治理指标"""
        # 计算各域健康分数
        for domain in GovernanceDomain:
            domain_rules = [
                r for r in self.governance_rules.values()
                if r.domain == domain
            ]
            
            if domain_rules:
                avg_effectiveness = sum(r.effectiveness_score for r in domain_rules) / len(domain_rules)
                self.domain_health_scores[domain.value] = avg_effectiveness
        
        # 更新整体治理指标
        self.governance_metrics['total_rules'] = len(self.governance_rules)
        self.governance_metrics['active_agents'] = len(self.agent_profiles)
        self.governance_metrics['recent_decisions'] = len([
            d for d in self.governance_decisions.values()
            if d.timestamp > datetime.now() - timedelta(hours=24)
        ])
        
        # 计算平均信任度和可靠性
        if self.agent_profiles:
            avg_trust = sum(p.trust_score for p in self.agent_profiles.values()) / len(self.agent_profiles)
            avg_reliability = sum(p.reliability_score for p in self.agent_profiles.values()) / len(self.agent_profiles)
            
            self.governance_metrics['avg_agent_trust'] = avg_trust
            self.governance_metrics['avg_agent_reliability'] = avg_reliability
    
    async def _check_system_health(self):
        """检查系统健康"""
        health_issues = []
        
        # 检查规则效果
        low_effectiveness_rules = [
            r for r in self.governance_rules.values()
            if r.effectiveness_score < 0.5
        ]
        
        if len(low_effectiveness_rules) > len(self.governance_rules) * 0.3:
            health_issues.append("大量规则效果不佳")
        
        # 检查智能体信任度
        low_trust_agents = [
            a for a in self.agent_profiles.values()
            if a.trust_score < 0.3
        ]
        
        if low_trust_agents:
            health_issues.append(f"{len(low_trust_agents)} 个智能体信任度过低")
        
        # 记录健康问题
        if health_issues:
            logger.warning(f"系统健康问题: {', '.join(health_issues)}")
    
    async def _process_pending_decisions(self):
        """处理待决策"""
        # 这里可以添加待决策队列的处理逻辑
        pass
    
    async def _monitor_agent_activity(self):
        """监控智能体活动"""
        current_time = datetime.now()
        inactive_threshold = timedelta(minutes=30)
        
        for agent_id, profile in self.agent_profiles.items():
            if profile.last_active:
                inactive_time = current_time - profile.last_active
                if inactive_time > inactive_threshold:
                    # 降低不活跃智能体的信任度
                    profile.trust_score = max(0.0, profile.trust_score - 0.01)
    
    async def _update_collaboration_graph(self):
        """更新协作图"""
        # 基于最近的协作历史更新图
        recent_collaborations = [
            h for profile in self.agent_profiles.values()
            for h in profile.collaboration_history
            if h['timestamp'] > datetime.now() - timedelta(hours=24)
        ]
        
        # 清空现有边
        self.collaboration_graph.clear_edges()
        for agent_id in self.agent_profiles:
            self.collaboration_graph.add_node(agent_id)
        
        # 添加协作边
        collaboration_counts = defaultdict(int)
        for collab in recent_collaborations:
            # 这里需要从collab中提取协作的智能体对
            # 简化实现
            pass
        
        # 添加边到图中
        for (agent1, agent2), count in collaboration_counts.items():
            if agent1 in self.agent_profiles and agent2 in self.agent_profiles:
                self.collaboration_graph.add_edge(agent1, agent2, weight=count)
    
    async def _detect_anomalous_behavior(self):
        """检测异常行为"""
        # 简化实现：检测决策频率异常
        for agent_id, profile in self.agent_profiles.items():
            recent_decisions = len([
                d for d in profile.collaboration_history
                if d['timestamp'] > datetime.now() - timedelta(hours=1)
            ])
            
            if recent_decisions > 100:  # 异常高频
                logger.warning(f"检测到异常高频决策: {agent_id}")
                # 可能需要限制该智能体的权限
    
    async def _analyze_decision_patterns(self):
        """分析决策模式"""
        # 分析各域的决策趋势
        for domain in GovernanceDomain:
            domain_decisions = [
                d for d in self.governance_decisions.values()
                if d.domain == domain and 
                   d.timestamp > datetime.now() - timedelta(days=7)
            ]
            
            if domain_decisions:
                permit_rate = sum(1 for d in domain_decisions if d.decision == 'permit') / len(domain_decisions)
                
                # 记录趋势
                self.governance_metrics[f'{domain.value}_permit_rate'] = permit_rate
    
    async def _collect_decision_feedback(self):
        """收集决策反馈"""
        # 这里可以实现自动反馈收集机制
        # 例如基于系统性能、用户满意度等
        pass
    
    async def _optimize_decision_strategies(self):
        """优化决策策略"""
        # 基于反馈调整决策策略
        for rule in self.governance_rules.values():
            # 获取相关决策的反馈
            related_decisions = [
                d for d in self.governance_decisions.values()
                if rule.rule_id in d.rules_applied and
                   d.timestamp > datetime.now() - timedelta(days=7)
            ]
            
            if related_decisions:
                feedback_scores = [d.feedback_score or 0.5 for d in related_decisions]
                avg_feedback = sum(feedback_scores) / len(feedback_scores)
                
                # 更新规则效果评分
                rule.effectiveness_score = rule.effectiveness_score * (1 - self.learning_rate) + avg_feedback * self.learning_rate
    
    async def _cleanup_expired_rules(self):
        """清理过期规则"""
        expiration_threshold = datetime.now() - timedelta(days=30)
        
        expired_rules = [
            rule_id for rule_id, rule in self.governance_rules.items()
            if not rule.enabled and rule.last_modified < expiration_threshold
        ]
        
        for rule_id in expired_rules:
            del self.governance_rules[rule_id]
            logger.info(f"清理过期规则: {rule_id}")
    
    async def get_governance_status(self) -> Dict[str, Any]:
        """获取治理状态"""
        return {
            'governance_rules': len(self.governance_rules),
            'registered_agents': len(self.agent_profiles),
            'total_decisions': len(self.governance_decisions),
            'domain_health_scores': dict(self.domain_health_scores),
            'governance_metrics': dict(self.governance_metrics),
            'recent_evolution_cycles': len(self.rule_evolution_history)
        }
    
    async def shutdown(self):
        """优雅关闭"""
        logger.info("正在关闭Meta-Agent治理层...")
        
        # 保存治理状态
        await self._save_governance_state()
        
        logger.info("Meta-Agent治理层已关闭")
    
    async def _save_governance_state(self):
        """保存治理状态"""
        state_file = PROJECT_ROOT / ".iflow" / "data" / "governance_state_v11.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'governance_rules': {
                    rule_id: asdict(rule) for rule_id, rule in self.governance_rules.items()
                },
                'agent_profiles': {
                    agent_id: asdict(profile) for agent_id, profile in self.agent_profiles.items()
                },
                'governance_metrics': dict(self.governance_metrics),
                'domain_health_scores': dict(self.domain_health_scores)
            }
            
            # 处理不可序列化的对象
            state['agent_profiles'] = {
                agent_id: {
                    **profile,
                    'permissions': {
                        domain.value: perm.value for domain, perm in profile.permissions.items()
                    }
                }
                for agent_id, profile in self.agent_profiles.items()
            }
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info("治理状态保存成功")
            
        except Exception as e:
            logger.error(f"保存治理状态失败: {e}")

# 全局实例
_meta_governor: Optional[MetaAgentGovernorV11] = None

async def get_meta_governor() -> MetaAgentGovernorV11:
    """获取Meta-Agent治理层实例"""
    global _meta_governor
    if _meta_governor is None:
        _meta_governor = MetaAgentGovernorV11()
        await _meta_governor.initialize()
    return _meta_governor

async def govern_agent_action(agent_id: str,
                             action: str,
                             domain: GovernanceDomain,
                             context: Dict[str, Any]) -> Dict[str, Any]:
    """治理智能体行动的便捷函数"""
    governor = await get_meta_governor()
    return await governor.govern_agent_action(agent_id, action, domain, context)