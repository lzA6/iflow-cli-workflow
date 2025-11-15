#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 智能体生命周期管理器 V9 (Agent Lifecycle Manager V9)
全面的智能体生命周期管理，包括创建、部署、监控和销毁

V9核心功能：
1. 智能体自动发现和注册
2. 生命周期状态管理
3. 健康检查和自动恢复
4. 资源分配和优化
5. 版本管理和升级
6. 性能监控和调优
7. 故障检测和处理
8. 优雅关闭和重启
"""

import asyncio
import json
import logging
import time
import uuid
import psutil
import threading
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import weakref

from unified_agent_template_v9 import BaseAgentV9, AgentConfig, AgentStatus

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LifecycleState(Enum):
    """生命周期状态"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    IDLE = "idle"
    BUSY = "busy"
    SUSPENDED = "suspended"
    ERROR = "error"
    RECOVERING = "recovering"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"

class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class AgentInstance:
    """智能体实例"""
    agent_id: str
    agent: BaseAgentV9
    config: AgentConfig
    state: LifecycleState = LifecycleState.INITIALIZING
    health_status: HealthStatus = HealthStatus.UNKNOWN
    created_time: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    restart_count: int = 0
    max_restarts: int = 3
    resource_usage: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class LifecyclePolicy:
    """生命周期策略"""
    max_idle_time: int = 300  # 最大空闲时间（秒）
    health_check_interval: int = 30  # 健康检查间隔（秒）
    auto_restart: bool = True  # 自动重启
    max_restarts: int = 3  # 最大重启次数
    resource_limits: Dict[str, float] = field(default_factory=dict)
    performance_thresholds: Dict[str, float] = field(default_factory=dict)

class AgentLifecycleManagerV9:
    """智能体生命周期管理器 V9"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 智能体实例管理
        self.agent_instances: Dict[str, AgentInstance] = {}
        self.agent_registry: Dict[str, AgentConfig] = {}
        
        # 生命周期策略
        self.default_policy = LifecyclePolicy(
            max_idle_time=self.config.get('max_idle_time', 300),
            health_check_interval=self.config.get('health_check_interval', 30),
            auto_restart=self.config.get('auto_restart', True),
            max_restarts=self.config.get('max_restarts', 3),
            resource_limits=self.config.get('resource_limits', {
                'max_memory_mb': 1024,
                'max_cpu_percent': 80
            }),
            performance_thresholds=self.config.get('performance_thresholds', {
                'max_response_time': 5.0,
                'min_success_rate': 0.9
            })
        )
        
        # 事件系统
        self.event_listeners: Dict[str, List[Callable]] = defaultdict(list)
        
        # 监控指标
        self.metrics = {
            'total_agents': 0,
            'running_agents': 0,
            'idle_agents': 0,
            'error_agents': 0,
            'total_restarts': 0,
            'avg_uptime': 0.0,
            'resource_usage': {
                'total_memory_mb': 0.0,
                'total_cpu_percent': 0.0
            }
        }
        
        # 后台任务
        self.background_tasks = set()
        self._start_background_tasks()
        
        logger.info("智能体生命周期管理器V9初始化完成")
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 健康检查任务
        health_task = asyncio.create_task(self._health_check_loop())
        self.background_tasks.add(health_task)
        
        # 资源监控任务
        resource_task = asyncio.create_task(self._resource_monitoring_loop())
        self.background_tasks.add(resource_task)
        
        # 生命周期管理任务
        lifecycle_task = asyncio.create_task(self._lifecycle_management_loop())
        self.background_tasks.add(lifecycle_task)
        
        # 指标更新任务
        metrics_task = asyncio.create_task(self._metrics_update_loop())
        self.background_tasks.add(metrics_task)
    
    async def register_agent(self, agent: BaseAgentV9, policy: Optional[LifecyclePolicy] = None) -> bool:
        """注册智能体"""
        try:
            agent_id = agent.agent_id
            
            # 检查是否已注册
            if agent_id in self.agent_instances:
                logger.warning(f"智能体 {agent_id} 已注册，更新实例")
                await self.unregister_agent(agent_id)
            
            # 创建实例
            instance = AgentInstance(
                agent_id=agent_id,
                agent=agent,
                config=agent.config,
                max_restarts=policy.max_restarts if policy else self.default_policy.max_restarts
            )
            
            # 存储实例
            self.agent_instances[agent_id] = instance
            self.agent_registry[agent_id] = agent.config
            
            # 启动智能体
            await agent.start()
            
            # 更新状态
            instance.state = LifecycleState.RUNNING
            instance.health_status = HealthStatus.HEALTHY
            
            # 更新指标
            self.metrics['total_agents'] += 1
            self.metrics['running_agents'] += 1
            
            # 触发事件
            await self._emit_event('agent_registered', {
                'agent_id': agent_id,
                'agent_name': agent.config.name,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"智能体 {agent.config.name} ({agent_id}) 注册成功")
            return True
            
        except Exception as e:
            logger.error(f"注册智能体失败: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """注销智能体"""
        try:
            if agent_id not in self.agent_instances:
                logger.warning(f"智能体 {agent_id} 未注册")
                return False
            
            instance = self.agent_instances[agent_id]
            
            # 优雅关闭
            await self._shutdown_agent(instance)
            
            # 清理资源
            del self.agent_instances[agent_id]
            del self.agent_registry[agent_id]
            
            # 更新指标
            self.metrics['total_agents'] -= 1
            if instance.state == LifecycleState.RUNNING:
                self.metrics['running_agents'] -= 1
            elif instance.state == LifecycleState.IDLE:
                self.metrics['idle_agents'] -= 1
            elif instance.state == LifecycleState.ERROR:
                self.metrics['error_agents'] -= 1
            
            # 触发事件
            await self._emit_event('agent_unregistered', {
                'agent_id': agent_id,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"智能体 {agent_id} 注销成功")
            return True
            
        except Exception as e:
            logger.error(f"注销智能体失败: {e}")
            return False
    
    async def _shutdown_agent(self, instance: AgentInstance):
        """关闭智能体"""
        try:
            instance.state = LifecycleState.SHUTTING_DOWN
            
            # 关闭智能体
            if hasattr(instance.agent, 'shutdown'):
                await instance.agent.shutdown()
            
            instance.state = LifecycleState.TERMINATED
            instance.health_status = HealthStatus.UNKNOWN
            
        except Exception as e:
            logger.error(f"关闭智能体失败: {e}")
            instance.state = LifecycleState.ERROR
            instance.health_status = HealthStatus.CRITICAL
    
    async def restart_agent(self, agent_id: str) -> bool:
        """重启智能体"""
        try:
            if agent_id not in self.agent_instances:
                logger.error(f"智能体 {agent_id} 不存在")
                return False
            
            instance = self.agent_instances[agent_id]
            
            # 检查重启次数
            if instance.restart_count >= instance.max_restarts:
                logger.error(f"智能体 {agent_id} 重启次数已达上限")
                return False
            
            instance.state = LifecycleState.RECOVERING
            instance.restart_count += 1
            self.metrics['total_restarts'] += 1
            
            # 记录重启原因
            instance.error_history.append({
                'timestamp': datetime.now().isoformat(),
                'reason': 'manual_restart',
                'restart_count': instance.restart_count
            })
            
            # 关闭旧实例
            await self._shutdown_agent(instance)
            
            # 创建新实例
            new_agent = self._create_agent_instance(instance.config)
            if new_agent:
                instance.agent = new_agent
                instance.state = LifecycleState.INITIALIZING
                
                # 启动新实例
                await new_agent.start()
                
                # 更新状态
                instance.state = LifecycleState.RUNNING
                instance.health_status = HealthStatus.HEALTHY
                instance.last_heartbeat = datetime.now()
                
                # 触发事件
                await self._emit_event('agent_restarted', {
                    'agent_id': agent_id,
                    'restart_count': instance.restart_count,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"智能体 {agent_id} 重启成功")
                return True
            else:
                logger.error(f"智能体 {agent_id} 重启失败，无法创建新实例")
                return False
                
        except Exception as e:
            logger.error(f"重启智能体失败: {e}")
            return False
    
    def _create_agent_instance(self, config: AgentConfig) -> Optional[BaseAgentV9]:
        """创建智能体实例（简化实现）"""
        # 这里应该根据配置创建实际的智能体实例
        # 简化实现返回None
        return None
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                current_time = datetime.now()
                
                for agent_id, instance in self.agent_instances.items():
                    # 检查心跳超时
                    if (current_time - instance.last_heartbeat).total_seconds() > 120:
                        instance.health_status = HealthStatus.CRITICAL
                        instance.state = LifecycleState.ERROR
                        
                        # 自动重启
                        if self.default_policy.auto_restart:
                            await self.restart_agent(agent_id)
                    
                    # 检查资源使用
                    await self._check_resource_usage(instance)
                    
                    # 检查性能指标
                    await self._check_performance_metrics(instance)
                
                await asyncio.sleep(self.default_policy.health_check_interval)
                
            except Exception as e:
                logger.error(f"健康检查错误: {e}")
                await asyncio.sleep(self.default_policy.health_check_interval)
    
    async def _check_resource_usage(self, instance: AgentInstance):
        """检查资源使用"""
        try:
            # 获取进程信息（简化实现）
            process = psutil.Process()
            
            # 内存使用
            memory_mb = process.memory_info().rss / 1024 / 1024
            instance.resource_usage['memory_mb'] = memory_mb
            
            # CPU使用
            cpu_percent = process.cpu_percent()
            instance.resource_usage['cpu_percent'] = cpu_percent
            
            # 检查限制
            max_memory = self.default_policy.resource_limits.get('max_memory_mb', 1024)
            max_cpu = self.default_policy.resource_limits.get('max_cpu_percent', 80)
            
            if memory_mb > max_memory:
                instance.health_status = HealthStatus.WARNING
                logger.warning(f"智能体 {instance.agent_id} 内存使用过高: {memory_mb}MB")
            
            if cpu_percent > max_cpu:
                instance.health_status = HealthStatus.WARNING
                logger.warning(f"智能体 {instance.agent_id} CPU使用过高: {cpu_percent}%")
                
        except Exception as e:
            logger.error(f"检查资源使用失败: {e}")
    
    async def _check_performance_metrics(self, instance: AgentInstance):
        """检查性能指标"""
        try:
            if hasattr(instance.agent, 'get_performance_metrics'):
                metrics = instance.agent.get_performance_metrics()
                instance.performance_metrics = metrics
                
                # 检查响应时间
                max_response_time = self.default_policy.performance_thresholds.get('max_response_time', 5.0)
                avg_time = metrics.get('avg_execution_time', 0)
                
                if avg_time > max_response_time:
                    instance.health_status = HealthStatus.WARNING
                    logger.warning(f"智能体 {instance.agent_id} 响应时间过长: {avg_time}s")
                
                # 检查成功率
                min_success_rate = self.default_policy.performance_thresholds.get('min_success_rate', 0.9)
                success_rate = metrics.get('success_rate', 1.0)
                
                if success_rate < min_success_rate:
                    instance.health_status = HealthStatus.WARNING
                    logger.warning(f"智能体 {instance.agent_id} 成功率过低: {success_rate}")
                    
        except Exception as e:
            logger.error(f"检查性能指标失败: {e}")
    
    async def _resource_monitoring_loop(self):
        """资源监控循环"""
        while True:
            try:
                total_memory = 0.0
                total_cpu = 0.0
                
                for instance in self.agent_instances.values():
                    total_memory += instance.resource_usage.get('memory_mb', 0)
                    total_cpu += instance.resource_usage.get('cpu_percent', 0)
                
                self.metrics['resource_usage']['total_memory_mb'] = total_memory
                self.metrics['resource_usage']['total_cpu_percent'] = total_cpu
                
                await asyncio.sleep(60)  # 每分钟更新一次
                
            except Exception as e:
                logger.error(f"资源监控错误: {e}")
                await asyncio.sleep(60)
    
    async def _lifecycle_management_loop(self):
        """生命周期管理循环"""
        while True:
            try:
                current_time = datetime.now()
                
                for agent_id, instance in self.agent_instances.items():
                    # 检查空闲时间
                    if instance.state == LifecycleState.IDLE:
                        idle_time = (current_time - instance.last_heartbeat).total_seconds()
                        if idle_time > self.default_policy.max_idle_time:
                            # 可以选择暂停或终止空闲智能体
                            logger.info(f"智能体 {agent_id} 空闲时间过长，考虑暂停")
                    
                    # 更新状态
                    if hasattr(instance.agent, 'status'):
                        agent_status = instance.agent.status
                        
                        if agent_status == AgentStatus.BUSY:
                            if instance.state != LifecycleState.BUSY:
                                instance.state = LifecycleState.BUSY
                                self.metrics['idle_agents'] -= 1
                                self.metrics['busy_agents'] = self.metrics.get('busy_agents', 0) + 1
                        elif agent_status == AgentStatus.IDLE:
                            if instance.state != LifecycleState.IDLE:
                                instance.state = LifecycleState.IDLE
                                self.metrics['busy_agents'] = self.metrics.get('busy_agents', 0) - 1
                                self.metrics['idle_agents'] = self.metrics.get('idle_agents', 0) + 1
                
                await asyncio.sleep(30)  # 每30秒检查一次
                
            except Exception as e:
                logger.error(f"生命周期管理错误: {e}")
                await asyncio.sleep(30)
    
    async def _metrics_update_loop(self):
        """指标更新循环"""
        while True:
            try:
                # 更新平均运行时间
                total_uptime = 0
                running_count = 0
                
                for instance in self.agent_instances.values():
                    uptime = (datetime.now() - instance.created_time).total_seconds()
                    total_uptime += uptime
                    running_count += 1
                
                if running_count > 0:
                    self.metrics['avg_uptime'] = total_uptime / running_count
                
                await asyncio.sleep(300)  # 每5分钟更新一次
                
            except Exception as e:
                logger.error(f"指标更新错误: {e}")
                await asyncio.sleep(300)
    
    def add_event_listener(self, event_type: str, listener: Callable):
        """添加事件监听器"""
        self.event_listeners[event_type].append(listener)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """触发事件"""
        for listener in self.event_listeners[event_type]:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(data)
                else:
                    listener(data)
            except Exception as e:
                logger.error(f"事件监听器错误: {e}")
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取智能体状态"""
        if agent_id in self.agent_instances:
            instance = self.agent_instances[agent_id]
            
            return {
                'agent_id': agent_id,
                'state': instance.state.value,
                'health_status': instance.health_status.value,
                'created_time': instance.created_time.isoformat(),
                'last_heartbeat': instance.last_heartbeat.isoformat(),
                'restart_count': instance.restart_count,
                'resource_usage': instance.resource_usage,
                'performance_metrics': instance.performance_metrics
            }
        
        return None
    
    def get_all_agents_status(self) -> List[Dict[str, Any]]:
        """获取所有智能体状态"""
        return [self.get_agent_status(agent_id) for agent_id in self.agent_instances.keys()]
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取管理器指标"""
        return {
            **self.metrics,
            'agent_instances': len(self.agent_instances),
            'healthy_agents': sum(1 for instance in self.agent_instances.values() 
                                 if instance.health_status == HealthStatus.HEALTHY),
            'warning_agents': sum(1 for instance in self.agent_instances.values() 
                                 if instance.health_status == HealthStatus.WARNING),
            'critical_agents': sum(1 for instance in self.agent_instances.values() 
                                 if instance.health_status == HealthStatus.CRITICAL)
        }
    
    async def shutdown_all(self):
        """关闭所有智能体"""
        logger.info("开始关闭所有智能体...")
        
        for agent_id in list(self.agent_instances.keys()):
            await self.unregister_agent(agent_id)
        
        # 取消后台任务
        for task in self.background_tasks:
            task.cancel()
        
        logger.info("所有智能体已关闭")
    
    async def shutdown(self):
        """关闭管理器"""
        await self.shutdown_all()
        logger.info("智能体生命周期管理器V9已关闭")

# 全局生命周期管理器实例
agent_lifecycle_manager_v9 = AgentLifecycleManagerV9()

# 示例使用
async def main():
    """示例使用"""
    # 添加事件监听器
    def on_agent_registered(data):
        print(f"智能体注册事件: {data}")
    
    def on_agent_restarted(data):
        print(f"智能体重启事件: {data}")
    
    agent_lifecycle_manager_v9.add_event_listener('agent_registered', on_agent_registered)
    agent_lifecycle_manager_v9.add_event_listener('agent_restarted', on_agent_restarted)
    
    # 获取指标
    metrics = agent_lifecycle_manager_v9.get_metrics()
    print(f"生命周期管理器指标: {metrics}")
    
    # 模拟运行
    await asyncio.sleep(1)
    
    # 关闭管理器
    await agent_lifecycle_manager_v9.shutdown()

if __name__ == "__main__":
    asyncio.run(main())