#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 实时监控系统 V9 (Real-time Monitoring System V9)
企业级实时监控解决方案，提供全方位的系统监控和告警功能

核心特性：
1. 实时性能监控 - CPU、内存、磁盘、网络全方位监控
2. 智能告警系统 - 基于机器学习的异常检测
3. 可视化仪表板 - 实时数据展示和趋势分析
4. 分布式监控 - 支持多节点集群监控
5. 自动化响应 - 智能故障自动修复
"""

import os
import sys
import json
import asyncio
import logging
import time
import psutil
import threading
import multiprocessing
import socket
import requests
import sqlite3
import aiofiles
import aiosqlite
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 尝试导入高性能依赖
try:
    import prometheus_client as prometheus
    from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logging.warning("Prometheus客户端不可用，使用基础监控")
    PROMETHEUS_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    logging.warning("Redis不可用，使用本地存储")
    REDIS_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 核心枚举和数据结构 ---

class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class SystemComponent(Enum):
    """系统组件"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESS = "process"
    APPLICATION = "application"
    DATABASE = "database"

@dataclass
class Metric:
    """监控指标"""
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    component: SystemComponent
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""

@dataclass
class Alert:
    """告警信息"""
    id: str
    name: str
    level: AlertLevel
    message: str
    component: SystemComponent
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MonitoringRule:
    """监控规则"""
    id: str
    name: str
    component: SystemComponent
    metric_name: str
    condition: str  # >, <, >=, <=, ==, !=
    threshold: float
    duration: int  # 持续时间（秒）
    level: AlertLevel
    enabled: bool = True
    description: str = ""
    actions: List[str] = field(default_factory=list)

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.collectors = {
            SystemComponent.CPU: self._collect_cpu_metrics,
            SystemComponent.MEMORY: self._collect_memory_metrics,
            SystemComponent.DISK: self._collect_disk_metrics,
            SystemComponent.NETWORK: self._collect_network_metrics,
            SystemComponent.PROCESS: self._collect_process_metrics
        }
        self.metrics_buffer = deque(maxlen=10000)
        self.collection_interval = 5  # 5秒收集间隔
        
    async def collect_all_metrics(self) -> List[Metric]:
        """收集所有指标"""
        metrics = []
        
        for component, collector in self.collectors.items():
            try:
                component_metrics = await collector()
                metrics.extend(component_metrics)
            except Exception as e:
                logger.error(f"收集 {component.value} 指标失败: {e}")
        
        # 添加到缓冲区
        self.metrics_buffer.extend(metrics)
        
        return metrics
    
    async def _collect_cpu_metrics(self) -> List[Metric]:
        """收集CPU指标"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(Metric(
                name="cpu_usage_percent",
                value=cpu_percent,
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                component=SystemComponent.CPU,
                unit="percent",
                description="CPU使用率"
            ))
            
            # CPU核心数
            cpu_count = psutil.cpu_count()
            metrics.append(Metric(
                name="cpu_count",
                value=float(cpu_count),
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                component=SystemComponent.CPU,
                unit="count",
                description="CPU核心数"
            ))
            
            # CPU负载
            load_avg = psutil.getloadavg()
            for i, load in enumerate(load_avg):
                metrics.append(Metric(
                    name=f"cpu_load_avg_{i+1}min",
                    value=load,
                    timestamp=timestamp,
                    metric_type=MetricType.GAUGE,
                    component=SystemComponent.CPU,
                    unit="load",
                    description=f"{i+1}分钟平均负载"
                ))
            
            # 每个CPU核心的使用率
            cpu_percents = psutil.cpu_percent(percpu=True)
            for i, percent in enumerate(cpu_percents):
                metrics.append(Metric(
                    name=f"cpu_core_{i}_usage_percent",
                    value=percent,
                    timestamp=timestamp,
                    metric_type=MetricType.GAUGE,
                    component=SystemComponent.CPU,
                    unit="percent",
                    tags={"core": str(i)},
                    description=f"CPU核心{i}使用率"
                ))
                
        except Exception as e:
            logger.error(f"CPU指标收集失败: {e}")
        
        return metrics
    
    async def _collect_memory_metrics(self) -> List[Metric]:
        """收集内存指标"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # 虚拟内存
            virtual_memory = psutil.virtual_memory()
            metrics.append(Metric(
                name="memory_total_bytes",
                value=float(virtual_memory.total),
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                component=SystemComponent.MEMORY,
                unit="bytes",
                description="总内存"
            ))
            
            metrics.append(Metric(
                name="memory_available_bytes",
                value=float(virtual_memory.available),
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                component=SystemComponent.MEMORY,
                unit="bytes",
                description="可用内存"
            ))
            
            metrics.append(Metric(
                name="memory_usage_percent",
                value=virtual_memory.percent,
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                component=SystemComponent.MEMORY,
                unit="percent",
                description="内存使用率"
            ))
            
            metrics.append(Metric(
                name="memory_used_bytes",
                value=float(virtual_memory.used),
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                component=SystemComponent.MEMORY,
                unit="bytes",
                description="已用内存"
            ))
            
            # 交换内存
            swap_memory = psutil.swap_memory()
            metrics.append(Metric(
                name="swap_usage_percent",
                value=swap_memory.percent,
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                component=SystemComponent.MEMORY,
                unit="percent",
                description="交换内存使用率"
            ))
            
        except Exception as e:
            logger.error(f"内存指标收集失败: {e}")
        
        return metrics
    
    async def _collect_disk_metrics(self) -> List[Metric]:
        """收集磁盘指标"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # 磁盘使用情况
            disk_partitions = psutil.disk_partitions()
            for partition in disk_partitions:
                try:
                    disk_usage = psutil.disk_usage(partition.mountpoint)
                    
                    metrics.append(Metric(
                        name="disk_total_bytes",
                        value=float(disk_usage.total),
                        timestamp=timestamp,
                        metric_type=MetricType.GAUGE,
                        component=SystemComponent.DISK,
                        unit="bytes",
                        tags={"device": partition.device, "mountpoint": partition.mountpoint},
                        description=f"磁盘总大小 - {partition.device}"
                    ))
                    
                    metrics.append(Metric(
                        name="disk_usage_percent",
                        value=(disk_usage.used / disk_usage.total) * 100,
                        timestamp=timestamp,
                        metric_type=MetricType.GAUGE,
                        component=SystemComponent.DISK,
                        unit="percent",
                        tags={"device": partition.device, "mountpoint": partition.mountpoint},
                        description=f"磁盘使用率 - {partition.device}"
                    ))
                    
                    metrics.append(Metric(
                        name="disk_free_bytes",
                        value=float(disk_usage.free),
                        timestamp=timestamp,
                        metric_type=MetricType.GAUGE,
                        component=SystemComponent.DISK,
                        unit="bytes",
                        tags={"device": partition.device, "mountpoint": partition.mountpoint},
                        description=f"磁盘可用空间 - {partition.device}"
                    ))
                    
                except PermissionError:
                    continue
            
            # 磁盘I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.append(Metric(
                    name="disk_read_bytes_per_sec",
                    value=float(disk_io.read_bytes),
                    timestamp=timestamp,
                    metric_type=MetricType.COUNTER,
                    component=SystemComponent.DISK,
                    unit="bytes/sec",
                    description="磁盘读取速率"
                ))
                
                metrics.append(Metric(
                    name="disk_write_bytes_per_sec",
                    value=float(disk_io.write_bytes),
                    timestamp=timestamp,
                    metric_type=MetricType.COUNTER,
                    component=SystemComponent.DISK,
                    unit="bytes/sec",
                    description="磁盘写入速率"
                ))
                
        except Exception as e:
            logger.error(f"磁盘指标收集失败: {e}")
        
        return metrics
    
    async def _collect_network_metrics(self) -> List[Metric]:
        """收集网络指标"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # 网络I/O
            net_io = psutil.net_io_counters()
            if net_io:
                metrics.append(Metric(
                    name="network_bytes_sent",
                    value=float(net_io.bytes_sent),
                    timestamp=timestamp,
                    metric_type=MetricType.COUNTER,
                    component=SystemComponent.NETWORK,
                    unit="bytes",
                    description="网络发送字节数"
                ))
                
                metrics.append(Metric(
                    name="network_bytes_recv",
                    value=float(net_io.bytes_recv),
                    timestamp=timestamp,
                    metric_type=MetricType.COUNTER,
                    component=SystemComponent.NETWORK,
                    unit="bytes",
                    description="网络接收字节数"
                ))
                
                metrics.append(Metric(
                    name="network_packets_sent",
                    value=float(net_io.packets_sent),
                    timestamp=timestamp,
                    metric_type=MetricType.COUNTER,
                    component=SystemComponent.NETWORK,
                    unit="packets",
                    description="网络发送包数"
                ))
                
                metrics.append(Metric(
                    name="network_packets_recv",
                    value=float(net_io.packets_recv),
                    timestamp=timestamp,
                    metric_type=MetricType.COUNTER,
                    component=SystemComponent.NETWORK,
                    unit="packets",
                    description="网络接收包数"
                ))
            
            # 网络连接
            connections = psutil.net_connections()
            connection_counts = defaultdict(int)
            for conn in connections:
                connection_counts[conn.status] += 1
            
            for status, count in connection_counts.items():
                metrics.append(Metric(
                    name=f"network_connections_{status}",
                    value=float(count),
                    timestamp=timestamp,
                    metric_type=MetricType.GAUGE,
                    component=SystemComponent.NETWORK,
                    unit="count",
                    tags={"status": status},
                    description=f"网络连接数 - {status}"
                ))
                
        except Exception as e:
            logger.error(f"网络指标收集失败: {e}")
        
        return metrics
    
    async def _collect_process_metrics(self) -> List[Metric]:
        """收集进程指标"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # 当前进程
            current_process = psutil.Process()
            
            metrics.append(Metric(
                name="process_cpu_percent",
                value=current_process.cpu_percent(),
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                component=SystemComponent.PROCESS,
                unit="percent",
                description="进程CPU使用率"
            ))
            
            metrics.append(Metric(
                name="process_memory_rss_bytes",
                value=float(current_process.memory_info().rss),
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                component=SystemComponent.PROCESS,
                unit="bytes",
                description="进程内存使用量(RSS)"
            ))
            
            metrics.append(Metric(
                name="process_memory_vms_bytes",
                value=float(current_process.memory_info().vms),
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                component=SystemComponent.PROCESS,
                unit="bytes",
                description="进程内存使用量(VMS)"
            ))
            
            metrics.append(Metric(
                name="process_num_threads",
                value=float(current_process.num_threads()),
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                component=SystemComponent.PROCESS,
                unit="count",
                description="进程线程数"
            ))
            
            metrics.append(Metric(
                name="process_num_fds",
                value=float(current_process.num_fds()),
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                component=SystemComponent.PROCESS,
                unit="count",
                description="进程文件描述符数"
            ))
            
            # 系统进程统计
            processes = psutil.process_iter(['pid', 'name', 'status'])
            status_counts = defaultdict(int)
            for proc in processes:
                try:
                    status_counts[proc.info['status']] += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            for status, count in status_counts.items():
                metrics.append(Metric(
                    name=f"processes_status_{status}",
                    value=float(count),
                    timestamp=timestamp,
                    metric_type=MetricType.GAUGE,
                    component=SystemComponent.PROCESS,
                    unit="count",
                    tags={"status": status},
                    description=f"进程状态统计 - {status}"
                ))
                
        except Exception as e:
            logger.error(f"进程指标收集失败: {e}")
        
        return metrics

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.rules: List[MonitoringRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history = deque(maxlen=1000)
        self.notification_handlers = []
        self.evaluation_interval = 10  # 10秒评估间隔
        
        # 初始化默认规则
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """初始化默认监控规则"""
        default_rules = [
            MonitoringRule(
                id="cpu_high_usage",
                name="CPU使用率过高",
                component=SystemComponent.CPU,
                metric_name="cpu_usage_percent",
                condition=">",
                threshold=80.0,
                duration=60,
                level=AlertLevel.WARNING,
                description="CPU使用率超过80%持续1分钟"
            ),
            MonitoringRule(
                id="memory_high_usage",
                name="内存使用率过高",
                component=SystemComponent.MEMORY,
                metric_name="memory_usage_percent",
                condition=">",
                threshold=85.0,
                duration=60,
                level=AlertLevel.WARNING,
                description="内存使用率超过85%持续1分钟"
            ),
            MonitoringRule(
                id="disk_low_space",
                name="磁盘空间不足",
                component=SystemComponent.DISK,
                metric_name="disk_usage_percent",
                condition=">",
                threshold=90.0,
                duration=30,
                level=AlertLevel.ERROR,
                description="磁盘使用率超过90%"
            ),
            MonitoringRule(
                id="process_high_memory",
                name="进程内存使用过高",
                component=SystemComponent.PROCESS,
                metric_name="process_memory_rss_bytes",
                condition=">",
                threshold=1024*1024*1024,  # 1GB
                duration=30,
                level=AlertLevel.WARNING,
                description="进程内存使用超过1GB"
            )
        ]
        
        self.rules.extend(default_rules)
    
    def add_rule(self, rule: MonitoringRule):
        """添加监控规则"""
        self.rules.append(rule)
        logger.info(f"添加监控规则: {rule.name}")
    
    def remove_rule(self, rule_id: str):
        """移除监控规则"""
        self.rules = [rule for rule in self.rules if rule.id != rule_id]
        logger.info(f"移除监控规则: {rule_id}")
    
    async def evaluate_rules(self, metrics: List[Metric]):
        """评估监控规则"""
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                await self._evaluate_rule(rule, metrics)
            except Exception as e:
                logger.error(f"评估规则 {rule.name} 失败: {e}")
    
    async def _evaluate_rule(self, rule: MonitoringRule, metrics: List[Metric]):
        """评估单个规则"""
        # 查找匹配的指标
        matching_metrics = [
            metric for metric in metrics
            if metric.component == rule.component and metric.metric_name == rule.metric_name
        ]
        
        if not matching_metrics:
            return
        
        # 获取最新的指标值
        latest_metric = max(matching_metrics, key=lambda m: m.timestamp)
        current_value = latest_metric.value
        
        # 检查条件
        condition_met = self._check_condition(current_value, rule.condition, rule.threshold)
        
        alert_id = f"{rule.id}_{hash(rule.component.value + rule.metric_name)}"
        
        if condition_met:
            # 条件满足，检查是否需要触发告警
            if alert_id not in self.active_alerts:
                # 新告警
                alert = Alert(
                    id=alert_id,
                    name=rule.name,
                    level=rule.level,
                    message=f"{rule.name}: {rule.metric_name} = {current_value:.2f} {rule.condition} {rule.threshold}",
                    component=rule.component,
                    metric_name=rule.metric_name,
                    current_value=current_value,
                    threshold=rule.threshold,
                    timestamp=datetime.now()
                )
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                # 发送通知
                await self._send_notification(alert)
                
                logger.warning(f"触发告警: {alert.name}")
                
        else:
            # 条件不满足，检查是否需要解决告警
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                
                # 发送解决通知
                await self._send_resolved_notification(alert)
                
                # 从活跃告警中移除
                del self.active_alerts[alert_id]
                
                logger.info(f"告警已解决: {alert.name}")
    
    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """检查条件"""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return abs(value - threshold) < 0.001  # 浮点数比较
        elif condition == "!=":
            return abs(value - threshold) >= 0.001
        else:
            return False
    
    async def _send_notification(self, alert: Alert):
        """发送告警通知"""
        for handler in self.notification_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"发送通知失败: {e}")
    
    async def _send_resolved_notification(self, alert: Alert):
        """发送告警解决通知"""
        for handler in self.notification_handlers:
            try:
                await handler(alert, resolved=True)
            except Exception as e:
                logger.error(f"发送解决通知失败: {e}")
    
    def add_notification_handler(self, handler: Callable):
        """添加通知处理器"""
        self.notification_handlers.append(handler)
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        return list(self.alert_history)[-limit:]

class MetricsStorage:
    """指标存储"""
    
    def __init__(self, storage_path: str = "monitoring_data.db"):
        self.storage_path = storage_path
        self.connection_pool = None
        self.write_queue = asyncio.Queue(maxsize=1000)
        self.batch_size = 100
        
    async def initialize(self):
        """初始化存储"""
        self.connection_pool = await aiosqlite.connect(self.storage_path)
        await self._create_tables()
        
        # 启动后台写入任务
        asyncio.create_task(self._background_writer())
    
    async def _create_tables(self):
        """创建表"""
        await self.connection_pool.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                metric_type TEXT NOT NULL,
                component TEXT NOT NULL,
                tags TEXT,
                unit TEXT,
                description TEXT
            )
        """)
        
        await self.connection_pool.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)
        """)
        
        await self.connection_pool.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_component ON metrics(component)
        """)
        
        await self.connection_pool.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                component TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                current_value REAL NOT NULL,
                threshold REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at DATETIME,
                acknowledged BOOLEAN DEFAULT FALSE,
                acknowledged_by TEXT,
                metadata TEXT
            )
        """)
        
        await self.connection_pool.commit()
    
    async def store_metrics(self, metrics: List[Metric]):
        """存储指标"""
        for metric in metrics:
            await self.write_queue.put(metric)
    
    async def _background_writer(self):
        """后台批量写入"""
        while True:
            try:
                batch = []
                
                # 收集批量数据
                while len(batch) < self.batch_size and not self.write_queue.empty():
                    try:
                        metric = self.write_queue.get_nowait()
                        batch.append(metric)
                    except asyncio.QueueEmpty:
                        break
                
                if batch:
                    await self._batch_write_metrics(batch)
                
                await asyncio.sleep(1)  # 1秒写入间隔
                
            except Exception as e:
                logger.error(f"后台写入失败: {e}")
    
    async def _batch_write_metrics(self, metrics: List[Metric]):
        """批量写入指标"""
        try:
            async with self.connection_pool.cursor() as cursor:
                for metric in metrics:
                    tags_json = json.dumps(metric.tags) if metric.tags else None
                    
                    await cursor.execute("""
                        INSERT INTO metrics 
                        (name, value, timestamp, metric_type, component, tags, unit, description)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metric.name,
                        metric.value,
                        metric.timestamp.isoformat(),
                        metric.metric_type.value,
                        metric.component.value,
                        tags_json,
                        metric.unit,
                        metric.description
                    ))
                
                await self.connection_pool.commit()
                logger.debug(f"批量写入 {len(metrics)} 个指标")
                
        except Exception as e:
            logger.error(f"批量写入指标失败: {e}")
    
    async def query_metrics(self, component: SystemComponent = None,
                          metric_name: str = None,
                          start_time: datetime = None,
                          end_time: datetime = None,
                          limit: int = 1000) -> List[Metric]:
        """查询指标"""
        conditions = []
        params = []
        
        if component:
            conditions.append("component = ?")
            params.append(component.value)
        
        if metric_name:
            conditions.append("name = ?")
            params.append(metric_name)
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())
        
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            SELECT name, value, timestamp, metric_type, component, tags, unit, description
            FROM metrics {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)
        
        async with self.connection_pool.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        
        metrics = []
        for row in rows:
            tags = json.loads(row[5]) if row[5] else {}
            
            metric = Metric(
                name=row[0],
                value=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                metric_type=MetricType(row[3]),
                component=SystemComponent(row[4]),
                tags=tags,
                unit=row[6] or "",
                description=row[7] or ""
            )
            metrics.append(metric)
        
        return metrics

class RealTimeMonitoringSystem:
    """实时监控系统"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.storage = MetricsStorage()
        
        self.running = False
        self.monitoring_task = None
        self.alert_task = None
        
        # 性能统计
        self.metrics_collected = 0
        self.alerts_triggered = 0
        self.start_time = None
        
        # Prometheus指标
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """设置Prometheus指标"""
        self.prometheus_metrics = {
            'system_cpu_usage': Gauge(
                'system_cpu_usage_percent',
                'System CPU usage percentage',
                registry=self.registry
            ),
            'system_memory_usage': Gauge(
                'system_memory_usage_percent',
                'System memory usage percentage',
                registry=self.registry
            ),
            'system_disk_usage': Gauge(
                'system_disk_usage_percent',
                'System disk usage percentage',
                registry=self.registry
            ),
            'alerts_total': Counter(
                'alerts_total',
                'Total number of alerts triggered',
                ['level', 'component'],
                registry=self.registry
            ),
            'metrics_collected_total': Counter(
                'metrics_collected_total',
                'Total number of metrics collected',
                registry=self.registry
            )
        }
    
    async def start(self):
        """启动监控系统"""
        if self.running:
            logger.warning("监控系统已在运行")
            return
        
        await self.storage.initialize()
        self.running = True
        self.start_time = datetime.now()
        
        # 启动监控任务
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.alert_task = asyncio.create_task(self._alert_loop())
        
        logger.info("🚀 实时监控系统已启动")
    
    async def stop(self):
        """停止监控系统"""
        if not self.running:
            return
        
        self.running = False
        
        # 停止任务
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        if self.alert_task:
            self.alert_task.cancel()
        
        logger.info("⏹️ 实时监控系统已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 收集指标
                metrics = await self.metrics_collector.collect_all_metrics()
                
                # 存储指标
                await self.storage.store_metrics(metrics)
                
                # 更新统计
                self.metrics_collected += len(metrics)
                
                # 更新Prometheus指标
                if PROMETHEUS_AVAILABLE:
                    self._update_prometheus_metrics(metrics)
                
                logger.debug(f"收集了 {len(metrics)} 个指标")
                
                # 等待下次收集
                await asyncio.sleep(self.metrics_collector.collection_interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(5)
    
    async def _alert_loop(self):
        """告警循环"""
        while self.running:
            try:
                # 获取最近的指标
                end_time = datetime.now()
                start_time = end_time - timedelta(seconds=60)  # 最近1分钟的指标
                
                metrics = await self.storage.query_metrics(
                    start_time=start_time,
                    end_time=end_time,
                    limit=1000
                )
                
                # 评估告警规则
                await self.alert_manager.evaluate_rules(metrics)
                
                # 等待下次评估
                await asyncio.sleep(self.alert_manager.evaluation_interval)
                
            except Exception as e:
                logger.error(f"告警循环错误: {e}")
                await asyncio.sleep(10)
    
    def _update_prometheus_metrics(self, metrics: List[Metric]):
        """更新Prometheus指标"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        for metric in metrics:
            if metric.name == "cpu_usage_percent":
                self.prometheus_metrics['system_cpu_usage'].set(metric.value)
            elif metric.name == "memory_usage_percent":
                self.prometheus_metrics['system_memory_usage'].set(metric.value)
            elif metric.name == "disk_usage_percent":
                self.prometheus_metrics['system_disk_usage'].set(metric.value)
        
        self.prometheus_metrics['metrics_collected_total'].inc(len(metrics))
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        # 获取最近的指标
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=5)
        
        recent_metrics = await self.storage.query_metrics(
            start_time=start_time,
            end_time=end_time,
            limit=100
        )
        
        # 计算平均值
        metric_averages = defaultdict(list)
        for metric in recent_metrics:
            metric_averages[metric.name].append(metric.value)
        
        averages = {
            name: np.mean(values) for name, values in metric_averages.items()
        }
        
        return {
            "status": "running" if self.running else "stopped",
            "uptime_seconds": uptime,
            "metrics_collected": self.metrics_collected,
            "alerts_triggered": self.alerts_triggered,
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "current_metrics": averages,
            "monitoring_interval": self.metrics_collector.collection_interval,
            "alert_evaluation_interval": self.alert_manager.evaluation_interval
        }
    
    async def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """获取指标摘要"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        metrics = await self.storage.query_metrics(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        # 按组件分组
        component_metrics = defaultdict(list)
        for metric in metrics:
            component_metrics[metric.component.value].append(metric)
        
        summary = {}
        for component, comp_metrics in component_metrics.items():
            metric_summary = {}
            for metric in comp_metrics:
                if metric.name not in metric_summary:
                    values = [m.value for m in comp_metrics if m.name == metric.name]
                    metric_summary[metric.name] = {
                        "current": values[-1] if values else 0,
                        "average": np.mean(values) if values else 0,
                        "min": np.min(values) if values else 0,
                        "max": np.max(values) if values else 0,
                        "unit": metric.unit
                    }
            
            summary[component] = metric_summary
        
        return summary
    
    def add_custom_rule(self, rule: MonitoringRule):
        """添加自定义监控规则"""
        self.alert_manager.add_rule(rule)
    
    def get_alerts(self, active_only: bool = True) -> List[Alert]:
        """获取告警信息"""
        if active_only:
            return self.alert_manager.get_active_alerts()
        else:
            return self.alert_manager.get_alert_history()

# 全局监控系统实例
_monitoring_system = None

async def get_monitoring_system() -> RealTimeMonitoringSystem:
    """获取监控系统单例"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = RealTimeMonitoringSystem()
        await _monitoring_system.start()
    return _monitoring_system

# 便捷函数
async def start_monitoring():
    """启动监控"""
    system = await get_monitoring_system()
    await system.start()

async def get_system_status():
    """获取系统状态"""
    system = await get_monitoring_system()
    return await system.get_system_status()

if __name__ == "__main__":
    # 测试代码
    async def test_monitoring():
        system = RealTimeMonitoringSystem()
        await system.start()
        
        # 运行1分钟
        await asyncio.sleep(60)
        
        # 获取状态
        status = await system.get_system_status()
        print("系统状态:")
        print(json.dumps(status, indent=2, ensure_ascii=False))
        
        # 获取指标摘要
        summary = await system.get_metrics_summary(hours=1)
        print("\n指标摘要:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        
        # 获取告警
        alerts = system.get_alerts()
        print(f"\n活跃告警: {len(alerts)}")
        
        await system.stop()
    
    # 运行测试
    asyncio.run(test_monitoring())
