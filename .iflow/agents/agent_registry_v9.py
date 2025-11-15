#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📋 智能体注册中心 V9 (Agent Registry V9)
统一的智能体管理、发现和协作平台

V9核心功能：
1. 智能体自动发现和注册
2. 能力匹配和任务分配
3. 协作网络管理
4. 性能监控和优化
5. 负载均衡和故障转移
6. 安全认证和授权
7. 实时状态同步
8. 智能体生命周期管理
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import weakref

from unified_agent_template_v9 import BaseAgentV9, AgentConfig, AgentCapability, AgentStatus, Task, TaskResult

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegistryEvent(Enum):
    """注册中心事件"""
    AGENT_REGISTERED = "agent_registered"
    AGENT_UNREGISTERED = "agent_unregistered"
    AGENT_STATUS_CHANGED = "agent_status_changed"
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    COLLABORATION_STARTED = "collaboration_started"
    COLLABORATION_ENDED = "collaboration_ended"

@dataclass
class AgentInfo:
    """智能体信息"""
    agent_id: str
    config: AgentConfig
    status: AgentStatus
    last_heartbeat: datetime
    performance_metrics: Dict[str, Any]
    active_tasks: Set[str] = field(default_factory=set)
    collaboration_partners: Set[str] = field(default_factory=set)
    registration_time: datetime = field(default_factory=datetime.now)

@dataclass
class TaskAssignment:
    """任务分配"""
    task_id: str
    agent_id: str
    assignment_time: datetime
    expected_completion: Optional[datetime] = None
    status: str = "assigned"
    retry_count: int = 0

class LoadBalancingStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CAPABILITY_MATCH = "capability_match"
    PERFORMANCE_BASED = "performance_based"

class AgentRegistryV9:
    """智能体注册中心 V9"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 智能体存储
        self.agents: Dict[str, AgentInfo] = {}
        self.capability_index: Dict[AgentCapability, Set[str]] = defaultdict(set)
        self.status_index: Dict[AgentStatus, Set[str]] = defaultdict(set)
        
        # 任务管理
        self.pending_tasks: Dict[str, Task] = {}
        self.task_assignments: Dict[str, TaskAssignment] = {}
        self.completed_tasks: List[TaskResult] = []
        
        # 负载均衡
        self.load_balancing_strategy = LoadBalancingStrategy(
            self.config.get('load_balancing_strategy', 'least_loaded')
        )
        self.round_robin_counter = 0
        
        # 事件系统
        self.event_listeners: Dict[RegistryEvent, List[Callable]] = defaultdict(list)
        
        # 监控和统计
        self.metrics = {
            'total_agents': 0,
            'active_agents': 0,
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_task_time': 0.0,
            'collaboration_count': 0
        }
        
        # 健康检查
        self.heartbeat_interval = self.config.get('heartbeat_interval', 30)
        self.agent_timeout = self.config.get('agent_timeout', 120)
        
        # 启动后台任务
        self.background_tasks = set()
        self._start_background_tasks()
        
        logger.info("智能体注册中心V9初始化完成")
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 健康检查任务
        health_task = asyncio.create_task(self._health_check_loop())
        self.background_tasks.add(health_task)
        
        # 任务分配任务
        assignment_task = asyncio.create_task(self._task_assignment_loop())
        self.background_tasks.add(assignment_task)
        
        # 指标更新任务
        metrics_task = asyncio.create_task(self._metrics_update_loop())
        self.background_tasks.add(metrics_task)
    
    async def register_agent(self, agent: BaseAgentV9) -> bool:
        """注册智能体"""
        try:
            agent_id = agent.agent_id
            
            # 检查是否已注册
            if agent_id in self.agents:
                logger.warning(f"智能体 {agent_id} 已注册，更新信息")
                await self.unregister_agent(agent_id)
            
            # 创建智能体信息
            agent_info = AgentInfo(
                agent_id=agent_id,
                config=agent.config,
                status=agent.status,
                last_heartbeat=datetime.now(),
                performance_metrics=agent.get_performance_metrics()
            )
            
            # 存储智能体信息
            self.agents[agent_id] = agent_info
            
            # 更新索引
            for capability in agent.get_capabilities():
                self.capability_index[capability].add(agent_id)
            self.status_index[agent.status].add(agent_id)
            
            # 设置事件监听
            agent.add_collaborator = lambda aid: self._on_collaboration_started(agent_id, aid)
            
            # 更新指标
            self.metrics['total_agents'] += 1
            if agent.status == AgentStatus.BUSY:
                self.metrics['active_agents'] += 1
            
            # 触发事件
            await self._emit_event(RegistryEvent.AGENT_REGISTERED, {
                'agent_id': agent_id,
                'agent_name': agent.config.name,
                'capabilities': [cap.value for cap in agent.get_capabilities()]
            })
            
            logger.info(f"智能体 {agent.config.name} ({agent_id}) 注册成功")
            return True
            
        except Exception as e:
            logger.error(f"注册智能体失败: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """注销智能体"""
        try:
            if agent_id not in self.agents:
                logger.warning(f"智能体 {agent_id} 未注册")
                return False
            
            agent_info = self.agents[agent_id]
            
            # 取消活跃任务
            for task_id in list(agent_info.active_tasks):
                await self._reassign_task(task_id)
            
            # 结束协作
            for partner_id in list(agent_info.collaboration_partners):
                await self._end_collaboration(agent_id, partner_id)
            
            # 从索引中移除
            for capability in agent_info.config.capabilities:
                self.capability_index[capability].discard(agent_id)
            self.status_index[agent_info.status].discard(agent_id)
            
            # 删除智能体
            del self.agents[agent_id]
            
            # 更新指标
            self.metrics['total_agents'] -= 1
            if agent_info.status == AgentStatus.BUSY:
                self.metrics['active_agents'] -= 1
            
            # 触发事件
            await self._emit_event(RegistryEvent.AGENT_UNREGISTERED, {
                'agent_id': agent_id,
                'agent_name': agent_info.config.name
            })
            
            logger.info(f"智能体 {agent_info.config.name} ({agent_id}) 注销成功")
            return True
            
        except Exception as e:
            logger.error(f"注销智能体失败: {e}")
            return False
    
    async def submit_task(self, task: Task) -> str:
        """提交任务"""
        task_id = task.task_id
        self.pending_tasks[task_id] = task
        self.metrics['total_tasks'] += 1
        
        logger.info(f"任务 {task_id} 已提交")
        return task_id
    
    async def _task_assignment_loop(self):
        """任务分配循环"""
        while True:
            try:
                # 获取待分配任务
                pending_task_ids = list(self.pending_tasks.keys())
                
                for task_id in pending_task_ids:
                    if task_id not in self.pending_tasks:
                        continue  # 任务可能已被分配
                    
                    task = self.pending_tasks[task_id]
                    
                    # 查找合适的智能体
                    suitable_agents = await self._find_suitable_agents(task)
                    
                    if suitable_agents:
                        # 选择最佳智能体
                        agent_id = await self._select_best_agent(suitable_agents, task)
                        
                        if agent_id:
                            # 分配任务
                            await self._assign_task(task_id, agent_id)
                        else:
                            logger.warning(f"任务 {task_id} 无可用智能体")
                    else:
                        logger.warning(f"任务 {task_id} 无合适智能体")
                
                await asyncio.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"任务分配循环错误: {e}")
                await asyncio.sleep(5)
    
    async def _find_suitable_agents(self, task: Task) -> List[str]:
        """查找合适的智能体"""
        suitable_agents = []
        
        # 根据任务类型匹配能力
        task_capability = self._map_task_to_capability(task.task_type)
        
        if task_capability and task_capability in self.capability_index:
            candidate_agents = self.capability_index[task_capability]
            
            for agent_id in candidate_agents:
                agent_info = self.agents.get(agent_id)
                if agent_info and agent_info.status in [AgentStatus.IDLE, AgentStatus.BUSY]:
                    # 检查负载
                    if len(agent_info.active_tasks) < agent_info.config.max_concurrent_tasks:
                        suitable_agents.append(agent_id)
        
        return suitable_agents
    
    def _map_task_to_capability(self, task_type: str) -> Optional[AgentCapability]:
        """映射任务类型到能力"""
        mapping = {
            'code_generation': AgentCapability.CODE_GENERATION,
            'data_analysis': AgentCapability.DATA_ANALYSIS,
            'system_design': AgentCapability.SYSTEM_DESIGN,
            'problem_solving': AgentCapability.PROBLEM_SOLVING,
            'communication': AgentCapability.COMMUNICATION,
            'learning': AgentCapability.LEARNING,
            'collaboration': AgentCapability.COLLABORATION,
            'optimization': AgentCapability.OPTIMIZATION
        }
        return mapping.get(task_type)
    
    async def _select_best_agent(self, suitable_agents: List[str], task: Task) -> Optional[str]:
        """选择最佳智能体"""
        if not suitable_agents:
            return None
        
        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(suitable_agents)
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._least_loaded_select(suitable_agents)
        elif self.load_balancing_strategy == LoadBalancingStrategy.CAPABILITY_MATCH:
            return self._capability_match_select(suitable_agents, task)
        elif self.load_balancing_strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            return self._performance_based_select(suitable_agents)
        else:
            return suitable_agents[0]
    
    def _round_robin_select(self, agents: List[str]) -> str:
        """轮询选择"""
        agent = agents[self.round_robin_counter % len(agents)]
        self.round_robin_counter += 1
        return agent
    
    def _least_loaded_select(self, agents: List[str]) -> str:
        """最少负载选择"""
        return min(agents, key=lambda aid: len(self.agents[aid].active_tasks))
    
    def _capability_match_select(self, agents: List[str], task: Task) -> str:
        """能力匹配选择"""
        # 简化实现：返回第一个
        return agents[0]
    
    def _performance_based_select(self, agents: List[str]) -> str:
        """基于性能选择"""
        def performance_score(agent_id: str) -> float:
            metrics = self.agents[agent_id].performance_metrics
            return (
                metrics.get('success_rate', 0) * 0.4 +
                (1.0 / (metrics.get('avg_execution_time', 1) + 0.1)) * 0.3 +
                (1.0 / (len(self.agents[agent_id].active_tasks) + 1)) * 0.3
            )
        
        return max(agents, key=performance_score)
    
    async def _assign_task(self, task_id: str, agent_id: str):
        """分配任务"""
        try:
            task = self.pending_tasks.pop(task_id)
            agent_info = self.agents[agent_id]
            
            # 创建任务分配
            assignment = TaskAssignment(
                task_id=task_id,
                agent_id=agent_id,
                assignment_time=datetime.now(),
                expected_completion=datetime.now() + timedelta(hours=1)
            )
            
            self.task_assignments[task_id] = assignment
            agent_info.active_tasks.add(task_id)
            
            # 更新智能体状态
            if agent_info.status == AgentStatus.IDLE:
                agent_info.status = AgentStatus.BUSY
                self.status_index[AgentStatus.IDLE].discard(agent_id)
                self.status_index[AgentStatus.BUSY].add(agent_id)
                self.metrics['active_agents'] += 1
            
            # 提交任务给智能体
            agent = self._get_agent_instance(agent_id)
            if agent:
                await agent.submit_task(task)
            
            # 触发事件
            await self._emit_event(RegistryEvent.TASK_ASSIGNED, {
                'task_id': task_id,
                'agent_id': agent_id,
                'task_type': task.task_type
            })
            
            logger.info(f"任务 {task_id} 分配给智能体 {agent_id}")
            
        except Exception as e:
            logger.error(f"分配任务失败: {e}")
            # 重新放回待分配队列
            self.pending_tasks[task_id] = task
    
    async def _reassign_task(self, task_id: str):
        """重新分配任务"""
        if task_id in self.task_assignments:
            assignment = self.task_assignments[task_id]
            old_agent_id = assignment.agent_id
            
            # 从旧智能体移除
            if old_agent_id in self.agents:
                self.agents[old_agent_id].active_tasks.discard(task_id)
            
            # 重新放回待分配队列
            if task_id not in self.pending_tasks:
                # 需要重新创建任务对象
                # 这里简化处理，直接标记为失败
                await self._mark_task_failed(task_id, "智能体离线")
    
    async def complete_task(self, task_id: str, result: TaskResult):
        """完成任务"""
        try:
            if task_id in self.task_assignments:
                assignment = self.task_assignments[task_id]
                agent_id = assignment.agent_id
                agent_info = self.agents.get(agent_id)
                
                # 从智能体移除任务
                if agent_info:
                    agent_info.active_tasks.discard(task_id)
                    
                    # 更新智能体状态
                    if not agent_info.active_tasks and agent_info.status == AgentStatus.BUSY:
                        agent_info.status = AgentStatus.IDLE
                        self.status_index[AgentStatus.BUSY].discard(agent_id)
                        self.status_index[AgentStatus.IDLE].add(agent_id)
                        self.metrics['active_agents'] -= 1
                
                # 移除任务分配
                del self.task_assignments[task_id]
                
                # 添加到已完成任务
                self.completed_tasks.append(result)
                self.metrics['completed_tasks'] += 1
                
                # 更新平均任务时间
                if result.execution_time > 0:
                    current_avg = self.metrics['avg_task_time']
                    total = self.metrics['completed_tasks']
                    self.metrics['avg_task_time'] = (
                        (current_avg * (total - 1) + result.execution_time) / total
                    )
                
                # 触发事件
                await self._emit_event(RegistryEvent.TASK_COMPLETED, {
                    'task_id': task_id,
                    'agent_id': agent_id,
                    'status': result.status,
                    'execution_time': result.execution_time
                })
                
                logger.info(f"任务 {task_id} 完成，状态: {result.status}")
            
        except Exception as e:
            logger.error(f"完成任务失败: {e}")
    
    async def _mark_task_failed(self, task_id: str, reason: str):
        """标记任务失败"""
        self.metrics['failed_tasks'] += 1
        
        # 创建失败结果
        result = TaskResult(
            task_id=task_id,
            agent_id="unknown",
            status="failed",
            result=reason,
            execution_time=0.0,
            timestamp=datetime.now()
        )
        
        self.completed_tasks.append(result)
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                current_time = datetime.now()
                timeout_agents = []
                
                for agent_id, agent_info in self.agents.items():
                    # 检查心跳超时
                    if (current_time - agent_info.last_heartbeat).total_seconds() > self.agent_timeout:
                        timeout_agents.append(agent_id)
                
                # 处理超时智能体
                for agent_id in timeout_agents:
                    logger.warning(f"智能体 {agent_id} 心跳超时，标记为离线")
                    await self.unregister_agent(agent_id)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"健康检查错误: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _metrics_update_loop(self):
        """指标更新循环"""
        while True:
            try:
                # 更新智能体性能指标
                for agent_id, agent_info in self.agents.items():
                    agent = self._get_agent_instance(agent_id)
                    if agent:
                        agent_info.performance_metrics = agent.get_performance_metrics()
                        agent_info.last_heartbeat = datetime.now()
                
                await asyncio.sleep(60)  # 每分钟更新一次
                
            except Exception as e:
                logger.error(f"指标更新错误: {e}")
                await asyncio.sleep(60)
    
    def _get_agent_instance(self, agent_id: str) -> Optional[BaseAgentV9]:
        """获取智能体实例（简化实现）"""
        # 这里应该从实际的智能体管理器获取实例
        # 简化实现返回None
        return None
    
    async def _on_collaboration_started(self, agent_id: str, partner_id: str):
        """协作开始事件"""
        if agent_id in self.agents and partner_id in self.agents:
            self.agents[agent_id].collaboration_partners.add(partner_id)
            self.agents[partner_id].collaboration_partners.add(agent_id)
            self.metrics['collaboration_count'] += 1
            
            await self._emit_event(RegistryEvent.COLLABORATION_STARTED, {
                'agent_id': agent_id,
                'partner_id': partner_id
            })
    
    async def _end_collaboration(self, agent_id: str, partner_id: str):
        """结束协作"""
        if agent_id in self.agents:
            self.agents[agent_id].collaboration_partners.discard(partner_id)
        if partner_id in self.agents:
            self.agents[partner_id].collaboration_partners.discard(agent_id)
        
        await self._emit_event(RegistryEvent.COLLABORATION_ENDED, {
            'agent_id': agent_id,
            'partner_id': partner_id
        })
    
    def add_event_listener(self, event: RegistryEvent, listener: Callable):
        """添加事件监听器"""
        self.event_listeners[event].append(listener)
    
    async def _emit_event(self, event: RegistryEvent, data: Dict[str, Any]):
        """触发事件"""
        for listener in self.event_listeners[event]:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(data)
                else:
                    listener(data)
            except Exception as e:
                logger.error(f"事件监听器错误: {e}")
    
    def get_agents_by_capability(self, capability: AgentCapability) -> List[AgentInfo]:
        """根据能力获取智能体"""
        agent_ids = self.capability_index.get(capability, set())
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]
    
    def get_agents_by_status(self, status: AgentStatus) -> List[AgentInfo]:
        """根据状态获取智能体"""
        agent_ids = self.status_index.get(status, set())
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        return {
            **self.metrics,
            'pending_tasks': len(self.pending_tasks),
            'active_assignments': len(self.task_assignments),
            'registered_agents': len(self.agents)
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'registry_version': 'V9',
            'total_agents': len(self.agents),
            'active_agents': len(self.get_agents_by_status(AgentStatus.BUSY)),
            'idle_agents': len(self.get_agents_by_status(AgentStatus.IDLE)),
            'pending_tasks': len(self.pending_tasks),
            'metrics': self.get_metrics(),
            'load_balancing_strategy': self.load_balancing_strategy.value
        }
    
    async def shutdown(self):
        """关闭注册中心"""
        # 取消后台任务
        for task in self.background_tasks:
            task.cancel()
        
        # 注销所有智能体
        for agent_id in list(self.agents.keys()):
            await self.unregister_agent(agent_id)
        
        logger.info("智能体注册中心V9已关闭")

# 全局注册中心实例
agent_registry_v9 = AgentRegistryV9()

# 示例使用
async def main():
    """示例使用"""
    # 添加事件监听器
    def on_agent_registered(data):
        print(f"智能体注册事件: {data}")
    
    def on_task_completed(data):
        print(f"任务完成事件: {data}")
    
    agent_registry_v9.add_event_listener(RegistryEvent.AGENT_REGISTERED, on_agent_registered)
    agent_registry_v9.add_event_listener(RegistryEvent.TASK_COMPLETED, on_task_completed)
    
    # 获取系统状态
    status = agent_registry_v9.get_system_status()
    print(f"注册中心状态: {json.dumps(status, indent=2, ensure_ascii=False)}")
    
    # 模拟运行
    await asyncio.sleep(1)
    
    # 关闭注册中心
    await agent_registry_v9.shutdown()

if __name__ == "__main__":
    asyncio.run(main())