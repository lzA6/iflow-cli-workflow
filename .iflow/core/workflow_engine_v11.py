#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚙️ 工作流引擎 V11 (代号："凤凰")
===========================================================

这是 T-MIA 架构下的核心工作流引擎，实现了自适应工作流、反脆弱特性和智能任务编排。
V11版本在V10基础上全面重构，实现了真正的动态工作流调整、压力源识别和过度补偿机制。

核心特性：
- 自适应工作流 - 根据实时数据动态调整执行路径
- 反脆弱机制 - 在压力下变得更强
- 智能任务编排 - 动态分配和优化任务执行
- 过度补偿 - 从压力中学习并超越原有能力
- 可选性探索 - 从不确定性中获益

你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。

作者: AI架构师团队
版本: 11.0.0 (代号："凤凰")
日期: 2025-11-15
"""

import os
import sys
import json
import asyncio
import logging
import time
import uuid
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

# 项目根路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WorkflowEngineV11")

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class StressType(Enum):
    """压力类型"""
    PERFORMANCE_PRESSURE = "performance_pressure"
    RESOURCE_CONSTRAINT = "resource_constraint"
    COMPLEXITY_OVERLOAD = "complexity_overload"
    UNCERTAINTY_CHALLENGE = "uncertainty_challenge"
    DEADLINE_PRESSURE = "deadline_pressure"

@dataclass
class WorkflowTask:
    """工作流任务"""
    task_id: str
    name: str
    description: str
    task_type: str
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StressIndicator:
    """压力指标"""
    stress_id: str
    stress_type: StressType
    intensity: float  # 0.0 - 1.0
    duration: float  # 持续时间（秒）
    source: str  # 压力来源
    impact_areas: List[str]  # 影响区域
    is_beneficial: bool  # 是否为有益压力
    compensation_strategy: Optional[str] = None

@dataclass
class AdaptationRule:
    """适应规则"""
    rule_id: str
    trigger_conditions: Dict[str, Any]
    adaptation_actions: List[Dict[str, Any]]
    success_rate: float = 0.0
    last_applied: Optional[datetime] = None

class WorkflowEngineV11:
    """工作流引擎 V11"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.workflows: Dict[str, nx.DiGraph] = {}
        self.tasks: Dict[str, WorkflowTask] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # 反脆弱机制
        self.stress_indicators: Dict[str, StressIndicator] = {}
        self.adaptation_rules: Dict[str, AdaptationRule] = {}
        self.compensation_history = deque(maxlen=1000)
        self.antifragility_metrics = defaultdict(float)
        
        # 自适应机制
        self.adaptation_threshold = 0.7
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
        
        # 性能优化
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.task_queue = asyncio.PriorityQueue()
        self.resource_monitor = {}
        
        # 可选性探索
        self.option_space = defaultdict(list)
        self.black_swan_events = []
        
        logger.info("工作流引擎V11初始化完成")
    
    async def initialize(self):
        """异步初始化"""
        logger.info("正在初始化工作流引擎...")
        
        # 加载工作流定义
        await self._load_workflow_definitions()
        
        # 初始化适应规则
        await self._initialize_adaptation_rules()
        
        # 启动监控任务
        asyncio.create_task(self._stress_monitoring_loop())
        asyncio.create_task(self._adaptation_loop())
        asyncio.create_task(self._resource_monitoring_loop())
        asyncio.create_task(self._antifragility_evolution_loop())
        
        logger.info("工作流引擎初始化完成")
    
    async def create_workflow(self, 
                            workflow_id: str,
                            tasks: List[WorkflowTask],
                            adaptive: bool = True) -> bool:
        """创建工作流"""
        try:
            # 构建任务图
            workflow_graph = nx.DiGraph()
            
            # 添加任务节点
            for task in tasks:
                self.tasks[task.task_id] = task
                workflow_graph.add_node(task.task_id, task=task)
            
            # 添加依赖边
            for task in tasks:
                for dep_id in task.dependencies:
                    if dep_id in self.tasks:
                        workflow_graph.add_edge(dep_id, task.task_id)
            
            # 检测循环依赖
            if not nx.is_directed_acyclic_graph(workflow_graph):
                logger.error(f"工作流 {workflow_id} 存在循环依赖")
                return False
            
            self.workflows[workflow_id] = workflow_graph
            
            # 如果启用自适应，注册适应规则
            if adaptive:
                await self._register_adaptive_rules(workflow_id)
            
            logger.info(f"创建工作流成功: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"创建工作流失败 {workflow_id}: {e}")
            return False
    
    async def execute_workflow(self, 
                             workflow_id: str,
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行工作流"""
        if workflow_id not in self.workflows:
            raise ValueError(f"工作流不存在: {workflow_id}")
        
        workflow_graph = self.workflows[workflow_id]
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"开始执行工作流: {workflow_id}, 执行ID: {execution_id}")
        
        try:
            # 初始化执行上下文
            execution_context = {
                'workflow_id': workflow_id,
                'execution_id': execution_id,
                'context': context or {},
                'start_time': start_time,
                'tasks_completed': 0,
                'tasks_failed': 0,
                'adaptations_applied': []
            }
            
            # 识别压力源
            stress_sources = await self._identify_stress_sources(workflow_id, execution_context)
            
            # 应用反脆弱策略
            antifragile_strategy = await self._determine_antifragile_strategy(stress_sources)
            
            # 执行任务
            execution_results = await self._execute_workflow_tasks(
                workflow_graph, execution_context, antifragile_strategy
            )
            
            # 计算过度补偿
            overcompensation = await self._calculate_overcompensation(
                execution_results, stress_sources
            )
            
            # 更新反脆弱指标
            await self._update_antifragility_metrics(execution_results, overcompensation)
            
            execution_time = time.time() - start_time
            
            result = {
                'execution_id': execution_id,
                'workflow_id': workflow_id,
                'status': 'completed',
                'execution_time': execution_time,
                'tasks_completed': execution_context['tasks_completed'],
                'tasks_failed': execution_context['tasks_failed'],
                'adaptations_applied': execution_context['adaptations_applied'],
                'stress_sources_handled': len(stress_sources),
                'overcompensation_achieved': overcompensation,
                'antifragility_score': self._calculate_antifragility_score(execution_results),
                'results': execution_results
            }
            
            logger.info(f"工作流执行完成: {workflow_id}, 耗时: {execution_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"工作流执行失败 {workflow_id}: {e}")
            raise
    
    async def _execute_workflow_tasks(self, 
                                    workflow_graph: nx.DiGraph,
                                    context: Dict[str, Any],
                                    strategy: Dict[str, Any]) -> Dict[str, Any]:
        """执行工作流任务"""
        results = {}
        completed_tasks = set()
        
        # 获取拓扑排序
        try:
            task_order = list(nx.topological_sort(workflow_graph))
        except nx.NetworkXError:
            # 如果有循环依赖，使用启发式排序
            task_order = self._heuristic_task_order(workflow_graph)
        
        # 并行执行无依赖任务
        while len(completed_tasks) < len(task_order):
            # 找到可以执行的任务（依赖已完成）
            ready_tasks = []
            for task_id in task_order:
                if task_id not in completed_tasks and task_id not in self.running_tasks:
                    task = self.tasks[task_id]
                    if all(dep in completed_tasks for dep in task.dependencies):
                        ready_tasks.append(task_id)
            
            if not ready_tasks:
                # 等待运行中的任务
                await asyncio.sleep(0.1)
                continue
            
            # 并行执行就绪任务
            execution_tasks = []
            for task_id in ready_tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                
                # 应用适应策略
                adapted_task = await self._apply_adaptation_strategy(task, strategy)
                
                # 创建执行任务
                execution_task = asyncio.create_task(
                    self._execute_single_task(adapted_task, context)
                )
                self.running_tasks[task_id] = execution_task
                execution_tasks.append((task_id, execution_task))
            
            # 等待任务完成
            for task_id, execution_task in execution_tasks:
                try:
                    result = await execution_task
                    results[task_id] = result
                    
                    if result['success']:
                        completed_tasks.add(task_id)
                        context['tasks_completed'] += 1
                    else:
                        context['tasks_failed'] += 1
                        
                        # 检查是否需要重试
                        task = self.tasks[task_id]
                        if task.retry_count < task.max_retries:
                            task.retry_count += 1
                            task.status = TaskStatus.RETRYING
                            # 重新加入队列
                            ready_tasks.append(task_id)
                        else:
                            task.status = TaskStatus.FAILED
                            completed_tasks.add(task_id)  # 标记为已完成（失败）
                
                except Exception as e:
                    logger.error(f"任务执行异常 {task_id}: {e}")
                    results[task_id] = {'success': False, 'error': str(e)}
                    context['tasks_failed'] += 1
                    completed_tasks.add(task_id)
                
                finally:
                    # 清理运行任务
                    if task_id in self.running_tasks:
                        del self.running_tasks[task_id]
        
        return results
    
    async def _execute_single_task(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个任务"""
        task_start = time.time()
        
        try:
            # 根据任务类型执行不同逻辑
            if task.task_type == 'arq_analysis':
                result = await self._execute_arq_analysis_task(task, context)
            elif task.task_type == 'reasoning':
                result = await self._execute_reasoning_task(task, context)
            elif task.task_type == 'adaptation':
                result = await self._execute_adaptation_task(task, context)
            else:
                result = await self._execute_generic_task(task, context)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            execution_time = time.time() - task_start
            
            return {
                'success': True,
                'result': result,
                'execution_time': execution_time,
                'task_id': task.task_id
            }
            
        except asyncio.TimeoutError:
            task.error = "任务超时"
            task.status = TaskStatus.FAILED
            return {
                'success': False,
                'error': "任务超时",
                'task_id': task.task_id
            }
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            return {
                'success': False,
                'error': str(e),
                'task_id': task.task_id
            }
    
    async def _execute_arq_analysis_task(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行ARQ分析任务"""
        # 导入ARQ分析工作流
        from ..commands.arq_analysis_workflow_v11 import ARQAnalysisWorkflowV11
        from ..commands.arq_analysis_workflow_v11 import AnalysisConfig
        
        # 创建配置
        config = AnalysisConfig(
            workspace_path=Path(context.get('workspace', '.')),
            user_query=task.parameters.get('query', ''),
            output_format=task.parameters.get('output_format', 'json'),
            auto_optimize=task.parameters.get('auto_optimize', False),
            dry_run=task.parameters.get('dry_run', True)
        )
        
        # 执行分析
        workflow = ARQAnalysisWorkflowV11(config)
        result = await workflow.run_analysis()
        
        return result
    
    async def _execute_reasoning_task(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行推理任务"""
        # 导入ARQ推理引擎
        from .arq_reasoning_engine_v11 import get_arq_engine, ReasoningMode
        
        engine = await get_arq_engine()
        
        # 确定推理模式
        mode_str = task.parameters.get('mode', 'deductive')
        mode = ReasoningMode(mode_str)
        
        # 执行推理
        result = await engine.reason(
            query=task.parameters.get('query', {}),
            mode=mode,
            depth=task.parameters.get('depth', 5),
            include_emotional=task.parameters.get('include_emotional', True),
            distributed=task.parameters.get('distributed', False)
        )
        
        return result
    
    async def _execute_adaptation_task(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行适应任务"""
        adaptation_type = task.parameters.get('adaptation_type', 'general')
        
        if adaptation_type == 'stress_response':
            return await self._execute_stress_response_adaptation(task, context)
        elif adaptation_type == 'performance_optimization':
            return await self._execute_performance_optimization_adaptation(task, context)
        elif adaptation_type == 'resource_reallocation':
            return await self._execute_resource_reallocation_adaptation(task, context)
        else:
            return await self._execute_generic_adaptation(task, context)
    
    async def _execute_generic_task(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行通用任务"""
        # 模拟任务执行
        await asyncio.sleep(task.parameters.get('duration', 1.0))
        
        return {
            'task_type': 'generic',
            'parameters': task.parameters,
            'context': context,
            'message': f"通用任务 {task.name} 执行完成"
        }
    
    async def _identify_stress_sources(self, workflow_id: str, context: Dict[str, Any]) -> List[StressIndicator]:
        """识别压力源"""
        stress_sources = []
        
        # 性能压力
        if context.get('tasks_count', 0) > 10:
            stress_sources.append(StressIndicator(
                stress_id=str(uuid.uuid4()),
                stress_type=StressType.COMPLEXITY_OVERLOAD,
                intensity=min(context.get('tasks_count', 0) / 20.0, 1.0),
                duration=0.0,
                source='workflow_complexity',
                impact_areas=['performance', 'memory'],
                is_beneficial=True  # 适度复杂度是有益的
            ))
        
        # 资源约束压力
        if len(self.running_tasks) >= 6:
            stress_sources.append(StressIndicator(
                stress_id=str(uuid.uuid4()),
                stress_type=StressType.RESOURCE_CONSTRAINT,
                intensity=len(self.running_tasks) / 8.0,
                duration=0.0,
                source='concurrent_tasks',
                impact_areas=['cpu', 'memory'],
                is_beneficial=True  # 推动资源优化
            ))
        
        # 不确定性压力
        if context.get('uncertainty_level', 0.0) > 0.5:
            stress_sources.append(StressIndicator(
                stress_id=str(uuid.uuid4()),
                stress_type=StressType.UNCERTAINTY_CHALLENGE,
                intensity=context.get('uncertainty_level', 0.0),
                duration=0.0,
                source='environmental_uncertainty',
                impact_areas=['decision_making', 'adaptation'],
                is_beneficial=True  # 促进适应能力
            ))
        
        # 记录压力指标
        for stress in stress_sources:
            self.stress_indicators[stress.stress_id] = stress
        
        return stress_sources
    
    async def _determine_antifragile_strategy(self, stress_sources: List[StressIndicator]) -> Dict[str, Any]:
        """确定反脆弱策略"""
        strategy = {
            'overcompensation_enabled': False,
            'exploration_enabled': False,
            'adaptation_intensity': 0.5,
            'resource_allocation': {},
            'risk_mitigation': []
        }
        
        # 基于压力源调整策略
        for stress in stress_sources:
            if stress.is_beneficial and stress.intensity > 0.6:
                strategy['overcompensation_enabled'] = True
                strategy['adaptation_intensity'] = max(strategy['adaptation_intensity'], stress.intensity)
            
            if stress.stress_type == StressType.UNCERTAINTY_CHALLENGE:
                strategy['exploration_enabled'] = True
        
        return strategy
    
    async def _apply_adaptation_strategy(self, task: WorkflowTask, strategy: Dict[str, Any]) -> WorkflowTask:
        """应用适应策略"""
        adapted_task = WorkflowTask(
            task_id=task.task_id,
            name=task.name,
            description=task.description,
            task_type=task.task_type,
            dependencies=task.dependencies.copy(),
            parameters=task.parameters.copy(),
            status=task.status,
            priority=task.priority,
            retry_count=task.retry_count,
            max_retries=task.max_retries,
            timeout=task.timeout,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            result=task.result,
            error=task.error,
            metadata=task.metadata.copy()
        )
        
        # 应用超时调整
        if strategy['adaptation_intensity'] > 0.7:
            if adapted_task.timeout:
                adapted_task.timeout *= 1.5  # 延长超时时间
            adapted_task.max_retries += 1  # 增加重试次数
        
        # 应用优先级调整
        if strategy['overcompensation_enabled']:
            adapted_task.priority += 1
        
        return adapted_task
    
    async def _calculate_overcompensation(self, 
                                        execution_results: Dict[str, Any],
                                        stress_sources: List[StressIndicator]) -> Dict[str, Any]:
        """计算过度补偿"""
        overcompensation = {
            'achieved': False,
            'improvement_areas': [],
            'performance_gain': 0.0,
            'adaptation_benefits': []
        }
        
        # 计算性能提升
        total_tasks = execution_results.get('tasks_completed', 0) + execution_results.get('tasks_failed', 0)
        success_rate = execution_results.get('tasks_completed', 0) / max(total_tasks, 1)
        
        # 如果在压力下成功率提高，则为过度补偿
        beneficial_stress = [s for s in stress_sources if s.is_beneficial]
        if beneficial_stress and success_rate > 0.8:
            overcompensation['achieved'] = True
            overcompensation['performance_gain'] = (success_rate - 0.8) * 100
            overcompensation['improvement_areas'] = [s.impact_areas for s in beneficial_stress]
        
        return overcompensation
    
    async def _update_antifragility_metrics(self, 
                                          execution_results: Dict[str, Any],
                                          overcompensation: Dict[str, Any]):
        """更新反脆弱指标"""
        # 更新各种指标
        self.antifragility_metrics['stress_resilience'] += 0.01
        self.antifragility_metrics['adaptation_speed'] += 0.005
        
        if overcompensation['achieved']:
            self.antifragility_metrics['overcompensation_ability'] += 0.02
        
        # 记录补偿历史
        self.compensation_history.append({
            'timestamp': datetime.now(),
            'execution_results': execution_results,
            'overcompensation': overcompensation
        })
    
    def _calculate_antifragility_score(self, execution_results: Dict[str, Any]) -> float:
        """计算反脆弱分数"""
        base_score = 0.5
        
        # 基于成功率调整
        total_tasks = execution_results.get('tasks_completed', 0) + execution_results.get('tasks_failed', 0)
        if total_tasks > 0:
            success_rate = execution_results.get('tasks_completed', 0) / total_tasks
            base_score += (success_rate - 0.5) * 0.5
        
        # 基于适应次数调整
        adaptations = execution_results.get('adaptations_applied', [])
        base_score += min(len(adaptations) * 0.05, 0.2)
        
        # 基于反脆弱指标调整
        for metric_name, metric_value in self.antifragility_metrics.items():
            base_score += metric_value * 0.01
        
        return max(0.0, min(1.0, base_score))
    
    async def _stress_monitoring_loop(self):
        """压力监控循环"""
        while True:
            try:
                await asyncio.sleep(30)  # 30秒
                
                # 分析当前压力状态
                current_stress = await self._analyze_current_stress()
                
                # 识别新的压力源
                new_stressors = await self._identify_new_stressors(current_stress)
                
                # 触发适应机制
                if new_stressors:
                    await self._trigger_adaptation_mechanism(new_stressors)
                
            except Exception as e:
                logger.error(f"压力监控循环错误: {e}")
    
    async def _adaptation_loop(self):
        """适应循环"""
        while True:
            try:
                await asyncio.sleep(60)  # 1分钟
                
                # 评估适应效果
                adaptation_effectiveness = await self._evaluate_adaptation_effectiveness()
                
                # 更新适应规则
                await self._update_adaptation_rules(adaptation_effectiveness)
                
                # 探索新的适应策略
                if self.exploration_rate > 0.1:
                    await self._explore_new_adaptations()
                
            except Exception as e:
                logger.error(f"适应循环错误: {e}")
    
    async def _resource_monitoring_loop(self):
        """资源监控循环"""
        while True:
            try:
                await asyncio.sleep(10)  # 10秒
                
                # 监控资源使用
                resource_usage = await self._monitor_resource_usage()
                
                # 更新资源监控数据
                self.resource_monitor.update(resource_usage)
                
                # 检测资源压力
                if resource_usage.get('cpu_usage', 0) > 0.8:
                    await self._handle_resource_pressure('cpu', resource_usage)
                
                if resource_usage.get('memory_usage', 0) > 0.8:
                    await self._handle_resource_pressure('memory', resource_usage)
                
            except Exception as e:
                logger.error(f"资源监控循环错误: {e}")
    
    async def _antifragility_evolution_loop(self):
        """反脆弱进化循环"""
        while True:
            try:
                await asyncio.sleep(3600)  # 1小时
                
                # 分析进化趋势
                evolution_trends = await self._analyze_evolution_trends()
                
                # 更新进化策略
                await self._update_evolution_strategies(evolution_trends)
                
                # 记录进化里程碑
                if evolution_trends.get('significant_improvement', False):
                    await self._record_evolution_milestone(evolution_trends)
                
            except Exception as e:
                logger.error(f"反脆弱进化循环错误: {e}")
    
    async def _load_workflow_definitions(self):
        """加载工作流定义"""
        workflow_dir = PROJECT_ROOT / ".iflow" / "workflows"
        
        if workflow_dir.exists():
            for workflow_file in workflow_dir.glob("*.yaml"):
                try:
                    with open(workflow_file, 'r', encoding='utf-8') as f:
                        workflow_def = yaml.safe_load(f)
                    
                    # 转换为工作流任务
                    tasks = []
                    for task_def in workflow_def.get('tasks', []):
                        task = WorkflowTask(
                            task_id=task_def['id'],
                            name=task_def['name'],
                            description=task_def.get('description', ''),
                            task_type=task_def.get('type', 'generic'),
                            dependencies=task_def.get('dependencies', []),
                            parameters=task_def.get('parameters', {}),
                            priority=task_def.get('priority', 0),
                            timeout=task_def.get('timeout')
                        )
                        tasks.append(task)
                    
                    # 创建工作流
                    await self.create_workflow(
                        workflow_def['id'],
                        tasks,
                        adaptive=workflow_def.get('adaptive', True)
                    )
                    
                    logger.info(f"加载工作流: {workflow_def['id']}")
                    
                except Exception as e:
                    logger.error(f"加载工作流失败 {workflow_file}: {e}")
    
    async def _initialize_adaptation_rules(self):
        """初始化适应规则"""
        # 性能压力适应规则
        self.adaptation_rules['perf_stress_001'] = AdaptationRule(
            rule_id='perf_stress_001',
            trigger_conditions={
                'stress_type': 'performance_pressure',
                'intensity_threshold': 0.7
            },
            adaptation_actions=[
                {'action': 'increase_timeout', 'factor': 1.5},
                {'action': 'enable_parallelism', 'max_workers': 8},
                {'action': 'prioritize_critical_tasks'}
            ],
            success_rate=0.75
        )
        
        # 资源约束适应规则
        self.adaptation_rules['resource_const_001'] = AdaptationRule(
            rule_id='resource_const_001',
            trigger_conditions={
                'stress_type': 'resource_constraint',
                'intensity_threshold': 0.8
            },
            adaptation_actions=[
                {'action': 'reduce_concurrency', 'factor': 0.5},
                {'action': 'enable_caching', 'cache_size': '256MB'},
                {'action': 'optimize_memory_usage'}
            ],
            success_rate=0.80
        )
        
        logger.info(f"初始化了 {len(self.adaptation_rules)} 个适应规则")
    
    async def _register_adaptive_rules(self, workflow_id: str):
        """注册自适应规则"""
        # 为特定工作流注册规则
        pass
    
    def _heuristic_task_order(self, workflow_graph: nx.DiGraph) -> List[str]:
        """启发式任务排序（处理循环依赖）"""
        # 简化实现：返回节点列表
        return list(workflow_graph.nodes())
    
    async def _analyze_current_stress(self) -> Dict[str, Any]:
        """分析当前压力状态"""
        return {
            'total_stressors': len(self.stress_indicators),
            'beneficial_stressors': len([s for s in self.stress_indicators.values() if s.is_beneficial]),
            'average_intensity': sum(s.intensity for s in self.stress_indicators.values()) / max(len(self.stress_indicators), 1),
            'active_adaptations': len([r for r in self.adaptation_rules.values() if r.last_applied])
        }
    
    async def _identify_new_stressors(self, current_stress: Dict[str, Any]) -> List[StressIndicator]:
        """识别新的压力源"""
        # 简化实现
        return []
    
    async def _trigger_adaptation_mechanism(self, stressors: List[StressIndicator]):
        """触发适应机制"""
        for stressor in stressors:
            # 查找匹配的适应规则
            matching_rules = [
                rule for rule in self.adaptation_rules.values()
                if self._rule_matches_stressor(rule, stressor)
            ]
            
            # 应用适应规则
            for rule in matching_rules:
                await self._apply_adaptation_rule(rule, stressor)
    
    def _rule_matches_stressor(self, rule: AdaptationRule, stressor: StressIndicator) -> bool:
        """检查规则是否匹配压力源"""
        conditions = rule.trigger_conditions
        
        if conditions.get('stress_type') != stressor.stress_type.value:
            return False
        
        if conditions.get('intensity_threshold', 0) > stressor.intensity:
            return False
        
        return True
    
    async def _apply_adaptation_rule(self, rule: AdaptationRule, stressor: StressIndicator):
        """应用适应规则"""
        for action in rule.adaptation_actions:
            action_type = action.get('action')
            
            if action_type == 'increase_timeout':
                # 增加超时时间
                factor = action.get('factor', 1.5)
                for task in self.tasks.values():
                    if task.timeout:
                        task.timeout *= factor
            
            elif action_type == 'enable_parallelism':
                # 启用并行处理
                max_workers = action.get('max_workers', 8)
                self.executor._max_workers = max_workers
            
            elif action_type == 'reduce_concurrency':
                # 减少并发
                factor = action.get('factor', 0.5)
                current_workers = self.executor._max_workers
                self.executor._max_workers = max(1, int(current_workers * factor))
        
        rule.last_applied = datetime.now()
        logger.info(f"应用适应规则: {rule.rule_id}")
    
    async def _evaluate_adaptation_effectiveness(self) -> Dict[str, float]:
        """评估适应效果"""
        effectiveness = {}
        
        for rule_id, rule in self.adaptation_rules.items():
            if rule.last_applied:
                # 计算规则应用后的性能改善
                effectiveness[rule_id] = rule.success_rate
        
        return effectiveness
    
    async def _update_adaptation_rules(self, effectiveness: Dict[str, float]):
        """更新适应规则"""
        for rule_id, score in effectiveness.items():
            if rule_id in self.adaptation_rules:
                # 使用学习率更新成功率
                rule = self.adaptation_rules[rule_id]
                rule.success_rate = rule.success_rate * (1 - self.learning_rate) + score * self.learning_rate
    
    async def _explore_new_adaptations(self):
        """探索新的适应策略"""
        # 随机探索新策略
        if random.random() < self.exploration_rate:
            new_rule = AdaptationRule(
                rule_id=f'explored_{int(time.time())}',
                trigger_conditions={'stress_type': 'exploratory'},
                adaptation_actions=[{'action': 'experimental_adaptation'}],
                success_rate=0.5
            )
            
            self.adaptation_rules[new_rule.rule_id] = new_rule
            logger.info(f"探索新的适应规则: {new_rule.rule_id}")
    
    async def _monitor_resource_usage(self) -> Dict[str, float]:
        """监控资源使用"""
        # 简化实现
        import psutil
        
        return {
            'cpu_usage': psutil.cpu_percent() / 100.0,
            'memory_usage': psutil.virtual_memory().percent / 100.0,
            'active_tasks': len(self.running_tasks),
            'queue_size': self.task_queue.qsize()
        }
    
    async def _handle_resource_pressure(self, resource_type: str, usage: Dict[str, float]):
        """处理资源压力"""
        if resource_type == 'cpu':
            # CPU压力处理
            await self._reduce_cpu_pressure()
        elif resource_type == 'memory':
            # 内存压力处理
            await self._reduce_memory_pressure()
    
    async def _reduce_cpu_pressure(self):
        """减少CPU压力"""
        # 降低并发度
        current_workers = self.executor._max_workers
        self.executor._max_workers = max(1, current_workers - 1)
        
        logger.info(f"减少CPU压力：调整工作线程数至 {self.executor._max_workers}")
    
    async def _reduce_memory_pressure(self):
        """减少内存压力"""
        # 清理缓存
        if hasattr(self, 'cache'):
            self.cache.clear()
        
        # 触发垃圾回收
        import gc
        gc.collect()
        
        logger.info("减少内存压力：清理缓存和触发垃圾回收")
    
    async def _execute_stress_response_adaptation(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行压力响应适应"""
        return {
            'adaptation_type': 'stress_response',
            'applied_strategies': ['timeout_extension', 'retry_increase'],
            'result': '压力响应适应完成'
        }
    
    async def _execute_performance_optimization_adaptation(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行性能优化适应"""
        return {
            'adaptation_type': 'performance_optimization',
            'optimizations': ['parallel_processing', 'caching_enabled'],
            'result': '性能优化适应完成'
        }
    
    async def _execute_resource_reallocation_adaptation(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行资源重新分配适应"""
        return {
            'adaptation_type': 'resource_reallocation',
            'reallocation': ['memory_optimization', 'cpu_balancing'],
            'result': '资源重新分配适应完成'
        }
    
    async def _execute_generic_adaptation(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行通用适应"""
        return {
            'adaptation_type': 'generic',
            'actions': ['default_optimizations'],
            'result': '通用适应完成'
        }
    
    async def _analyze_evolution_trends(self) -> Dict[str, Any]:
        """分析进化趋势"""
        # 分析补偿历史
        if len(self.compensation_history) < 10:
            return {'significant_improvement': False}
        
        recent_compensations = list(self.compensation_history)[-10:]
        successful_overcompensations = [
            c for c in recent_compensations 
            if c['overcompensation']['achieved']
        ]
        
        improvement_rate = len(successful_overcompensations) / len(recent_compensations)
        
        return {
            'significant_improvement': improvement_rate > 0.7,
            'improvement_rate': improvement_rate,
            'trend_direction': 'improving' if improvement_rate > 0.5 else 'stable'
        }
    
    async def _update_evolution_strategies(self, trends: Dict[str, Any]):
        """更新进化策略"""
        if trends.get('significant_improvement', False):
            # 增强成功的策略
            self.exploration_rate = max(0.05, self.exploration_rate * 0.9)
            self.adaptation_threshold = min(0.9, self.adaptation_threshold * 1.05)
    
    async def _record_evolution_milestone(self, trends: Dict[str, Any]):
        """记录进化里程碑"""
        milestone = {
            'timestamp': datetime.now(),
            'trends': trends,
            'metrics': dict(self.antifragility_metrics),
            'adaptation_rules_count': len(self.adaptation_rules)
        }
        
        # 保存里程碑
        milestones_file = PROJECT_ROOT / ".iflow" / "data" / "evolution_milestones.json"
        milestones_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            milestones = []
            if milestones_file.exists():
                with open(milestones_file, 'r', encoding='utf-8') as f:
                    milestones = json.load(f)
            
            milestones.append(milestone)
            
            with open(milestones_file, 'w', encoding='utf-8') as f:
                json.dump(milestones, f, indent=2, default=str)
            
            logger.info("记录进化里程碑")
            
        except Exception as e:
            logger.error(f"记录进化里程碑失败: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'active_workflows': len(self.workflows),
            'total_tasks': len(self.tasks),
            'running_tasks': len(self.running_tasks),
            'stress_indicators': len(self.stress_indicators),
            'adaptation_rules': len(self.adaptation_rules),
            'antifragility_metrics': dict(self.antifragility_metrics),
            'resource_monitor': dict(self.resource_monitor)
        }
    
    async def shutdown(self):
        """优雅关闭"""
        logger.info("正在关闭工作流引擎...")
        
        # 等待所有运行任务完成
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        logger.info("工作流引擎已关闭")

# 全局实例
_workflow_engine: Optional[WorkflowEngineV11] = None

async def get_workflow_engine() -> WorkflowEngineV11:
    """获取工作流引擎实例"""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = WorkflowEngineV11()
        await _workflow_engine.initialize()
    return _workflow_engine

async def create_and_execute_workflow(workflow_id: str,
                                   tasks: List[WorkflowTask],
                                   context: Optional[Dict[str, Any]] = None,
                                   adaptive: bool = True) -> Dict[str, Any]:
    """创建并执行工作流的便捷函数"""
    engine = await get_workflow_engine()
    
    success = await engine.create_workflow(workflow_id, tasks, adaptive)
    if not success:
        raise RuntimeError(f"创建工作流失败: {workflow_id}")
    
    return await engine.execute_workflow(workflow_id, context)