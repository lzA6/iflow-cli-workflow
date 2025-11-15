#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏥 系统健康监控 V1.0
System Health Monitor V1.0

你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import asyncio
import json
import logging
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from path_manager import get_path_manager
    from core.performance_optimizer import get_performance_optimizer
except ImportError as e:
    print(f"警告: 无法导入依赖模块: {e}")
    get_path_manager = None
    get_performance_optimizer = None

logger = logging.getLogger(__name__)

@dataclass
class HealthMetric:
    """健康指标"""
    name: str
    value: float
    unit: str
    threshold_warning: float
    threshold_critical: float
    status: str = "healthy"  # healthy, warning, critical
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""

@dataclass
class HealthAlert:
    """健康告警"""
    alert_id: str
    metric_name: str
    severity: str  # info, warning, critical
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class SystemHealthMonitor:
    """系统健康监控器"""
    
    def __init__(self):
        """初始化健康监控器"""
        self.path_manager = get_path_manager() if get_path_manager else None
        self.performance_optimizer = get_performance_optimizer() if get_performance_optimizer else None
        
        self.metrics = {}
        self.alerts = deque(maxlen=1000)
        self.monitoring_active = False
        self.alert_handlers = []
        
        # 监控配置
        self.monitoring_config = {
            'interval': 30,  # 监控间隔（秒）
            'retention_days': 7,  # 数据保留天数
            'alert_cooldown': 300,  # 告警冷却时间（秒）
            'email_enabled': False,
            'email_config': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'recipients': []
            }
        }
        
        # 健康指标定义
        self.health_metrics = {
            'cpu_usage': HealthMetric(
                name='cpu_usage',
                value=0.0,
                unit='%',
                threshold_warning=70.0,
                threshold_critical=90.0,
                description='CPU使用率'
            ),
            'memory_usage': HealthMetric(
                name='memory_usage',
                value=0.0,
                unit='%',
                threshold_warning=80.0,
                threshold_critical=95.0,
                description='内存使用率'
            ),
            'disk_usage': HealthMetric(
                name='disk_usage',
                value=0.0,
                unit='%',
                threshold_warning=85.0,
                threshold_critical=95.0,
                description='磁盘使用率'
            ),
            'response_time': HealthMetric(
                name='response_time',
                value=0.0,
                unit='ms',
                threshold_warning=2000.0,
                threshold_critical=5000.0,
                description='平均响应时间'
            ),
            'error_rate': HealthMetric(
                name='error_rate',
                value=0.0,
                unit='%',
                threshold_warning=5.0,
                threshold_critical=10.0,
                description='错误率'
            ),
            'cache_hit_rate': HealthMetric(
                name='cache_hit_rate',
                value=0.0,
                unit='%',
                threshold_warning=60.0,
                threshold_critical=40.0,
                description='缓存命中率'
            )
        }
        
        # 设置日志
        self._setup_logging()
        
        logger.info("🏥 系统健康监控器初始化完成")
    
    def _setup_logging(self):
        """设置日志"""
        if not self.path_manager:
            return
        
        log_dir = self.path_manager.log_dir
        log_dir.mkdir(exist_ok=True)
        
        # 健康监控日志
        health_log_file = log_dir / f"health_monitor_{datetime.now().strftime('%Y%m%d')}.log"
        
        # 配置健康监控日志
        health_logger = logging.getLogger("health_monitor")
        health_logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(health_log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        health_logger.addHandler(file_handler)
        self.health_logger = health_logger
    
    async def start_monitoring(self):
        """启动监控"""
        self.monitoring_active = True
        monitor_task = asyncio.create_task(self._monitoring_loop())
        
        # 注册告警处理器
        self.register_alert_handler(self._log_alert_handler)
        self.register_alert_handler(self._email_alert_handler)
        
        self.health_logger.info("🏥 系统健康监控已启动")
        return monitor_task
    
    async def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        self.health_logger.info("🏥 系统健康监控已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                await self._collect_health_metrics()
                await self._evaluate_health_status()
                await self._cleanup_old_data()
                
                await asyncio.sleep(self.monitoring_config['interval'])
                
            except Exception as e:
                self.health_logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(60)
    
    async def _collect_health_metrics(self):
        """收集健康指标"""
        timestamp = datetime.now()
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        self.health_metrics['cpu_usage'].value = cpu_percent
        self.health_metrics['cpu_usage'].timestamp = timestamp
        
        # 内存使用率
        memory = psutil.virtual_memory()
        self.health_metrics['memory_usage'].value = memory.percent
        self.health_metrics['memory_usage'].timestamp = timestamp
        
        # 磁盘使用率
        if self.path_manager:
            disk = psutil.disk_usage(str(self.path_manager.project_root))
            disk_percent = (disk.used / disk.total) * 100
            self.health_metrics['disk_usage'].value = disk_percent
            self.health_metrics['disk_usage'].timestamp = timestamp
        
        # 从性能优化器获取指标
        if self.performance_optimizer:
            perf_report = self.performance_optimizer.get_performance_report()
            if 'averages' in perf_report:
                averages = perf_report['averages']
                
                # 响应时间
                if 'response_time' in averages:
                    self.health_metrics['response_time'].value = averages['response_time'] * 1000  # 转换为毫秒
                    self.health_metrics['response_time'].timestamp = timestamp
                
                # 错误率
                if 'error_rate' in averages:
                    self.health_metrics['error_rate'].value = averages['error_rate'] * 100  # 转换为百分比
                    self.health_metrics['error_rate'].timestamp = timestamp
                
                # 缓存命中率
                if 'cache_hit_rate' in averages:
                    self.health_metrics['cache_hit_rate'].value = averages['cache_hit_rate'] * 100  # 转换为百分比
                    self.health_metrics['cache_hit_rate'].timestamp = timestamp
    
    async def _evaluate_health_status(self):
        """评估健康状态"""
        for metric_name, metric in self.health_metrics.items():
            old_status = metric.status
            
            # 确定状态
            if metric.value >= metric.threshold_critical:
                metric.status = "critical"
            elif metric.value >= metric.threshold_warning:
                metric.status = "warning"
            else:
                metric.status = "healthy"
            
            # 检查状态变化
            if old_status != metric.status:
                await self._handle_status_change(metric, old_status)
    
    async def _handle_status_change(self, metric: HealthMetric, old_status: str):
        """处理状态变化"""
        if metric.status in ["warning", "critical"]:
            alert = HealthAlert(
                alert_id=f"{metric.name}_{int(time.time())}",
                metric_name=metric.name,
                severity=metric.status,
                message=f"指标 {metric.name} 状态变为 {metric.status}: {metric.value:.2f}{metric.unit}",
                metadata={
                    'threshold_warning': metric.threshold_warning,
                    'threshold_critical': metric.threshold_critical,
                    'old_status': old_status
                }
            )
            
            self.alerts.append(alert)
            await self._process_alert(alert)
        
        elif old_status in ["warning", "critical"] and metric.status == "healthy":
            # 解除告警
            for alert in reversed(self.alerts):
                if (alert.metric_name == metric.name and 
                    not alert.resolved and 
                    alert.severity in ["warning", "critical"]):
                    alert.resolved = True
                    alert.resolved_timestamp = datetime.now()
                    
                    resolve_alert = HealthAlert(
                        alert_id=f"{metric.name}_resolved_{int(time.time())}",
                        metric_name=metric.name,
                        severity="info",
                        message=f"指标 {metric.name} 已恢复正常: {metric.value:.2f}{metric.unit}",
                        metadata={'resolved_alert_id': alert.alert_id}
                    )
                    
                    self.alerts.append(resolve_alert)
                    await self._process_alert(resolve_alert)
                    break
    
    async def _process_alert(self, alert: HealthAlert):
        """处理告警"""
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                self.health_logger.error(f"告警处理器错误: {e}")
    
    def register_alert_handler(self, handler: Callable):
        """注册告警处理器"""
        self.alert_handlers.append(handler)
    
    async def _log_alert_handler(self, alert: HealthAlert):
        """日志告警处理器"""
        severity_icons = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
        icon = severity_icons.get(alert.severity, "📢")
        
        if alert.resolved:
            self.health_logger.info(f"{icon} 告警解除: {alert.message}")
        else:
            self.health_logger.warning(f"{icon} 健康告警: {alert.message}")
    
    async def _email_alert_handler(self, alert: HealthAlert):
        """邮件告警处理器"""
        if not self.monitoring_config['email_enabled']:
            return
        
        # 只发送严重告警
        if alert.severity not in ["critical"]:
            return
        
        try:
            await self._send_email_alert(alert)
        except Exception as e:
            self.health_logger.error(f"发送邮件告警失败: {e}")
    
    async def _send_email_alert(self, alert: HealthAlert):
        """发送邮件告警"""
        config = self.monitoring_config['email_config']
        
        # 创建邮件
        msg = MimeMultipart()
        msg['From'] = config['username']
        msg['To'] = ', '.join(config['recipients'])
        
        if alert.resolved:
            msg['Subject'] = f"[已解除] 系统健康告警 - {alert.metric_name}"
            body = f"""
告警已解除:

指标: {alert.metric_name}
时间: {alert.timestamp}
消息: {alert.message}

系统状态已恢复正常。
"""
        else:
            msg['Subject'] = f"[{alert.severity.upper()}] 系统健康告警 - {alert.metric_name}"
            body = f"""
系统健康告警:

指标: {alert.metric_name}
严重级别: {alert.severity}
时间: {alert.timestamp}
消息: {alert.message}

请及时处理此告警。
"""
        
        msg.attach(MimeText(body, 'plain', 'utf-8'))
        
        # 发送邮件
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
        
        self.health_logger.info(f"📧 邮件告警已发送: {alert.alert_id}")
    
    async def _cleanup_old_data(self):
        """清理旧数据"""
        retention_date = datetime.now() - timedelta(days=self.monitoring_config['retention_days'])
        
        # 清理旧告警
        self.alerts = deque(
            (alert for alert in self.alerts if alert.timestamp > retention_date),
            maxlen=1000
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        overall_status = "healthy"
        critical_count = 0
        warning_count = 0
        
        for metric in self.health_metrics.values():
            if metric.status == "critical":
                critical_count += 1
                overall_status = "critical"
            elif metric.status == "warning":
                warning_count += 1
                if overall_status == "healthy":
                    overall_status = "warning"
        
        # 活跃告警
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'metrics': {name: asdict(metric) for name, metric in self.health_metrics.items()},
            'summary': {
                'total_metrics': len(self.health_metrics),
                'healthy_metrics': len([m for m in self.health_metrics.values() if m.status == "healthy"]),
                'warning_metrics': len([m for m in self.health_metrics.values() if m.status == "warning"]),
                'critical_metrics': len([m for m in self.health_metrics.values() if m.status == "critical"]),
                'active_alerts': len(active_alerts),
                'total_alerts': len(self.alerts)
            },
            'active_alerts': [asdict(alert) for alert in active_alerts[-10:]]  # 最近10个活跃告警
        }
    
    def configure_email_alerts(self, smtp_server: str, smtp_port: int, 
                              username: str, password: str, recipients: List[str]):
        """配置邮件告警"""
        self.monitoring_config['email_config'].update({
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'recipients': recipients
        })
        self.monitoring_config['email_enabled'] = True
        
        self.health_logger.info("📧 邮件告警配置完成")
    
    def set_monitoring_interval(self, interval: int):
        """设置监控间隔"""
        self.monitoring_config['interval'] = interval
        self.health_logger.info(f"⏱️ 监控间隔已设置为: {interval}秒")
    
    async def generate_health_report(self) -> str:
        """生成健康报告"""
        status = self.get_health_status()
        
        report = f"""
🏥 系统健康报告
生成时间: {status['timestamp']}
总体状态: {status['overall_status'].upper()}

📊 指标概览:
"""
        
        for metric_name, metric in status['metrics'].items():
            status_icon = {"healthy": "✅", "warning": "⚠️", "critical": "🚨"}[metric['status']]
            report += f"  {status_icon} {metric['description']}: {metric['value']:.2f}{metric['unit']} ({metric['status']})\n"
        
        report += f"""
📈 统计信息:
  总指标数: {status['summary']['total_metrics']}
  健康指标: {status['summary']['healthy_metrics']}
  警告指标: {status['summary']['warning_metrics']}
  严重指标: {status['summary']['critical_metrics']}
  活跃告警: {status['summary']['active_alerts']}
"""
        
        if status['active_alerts']:
            report += "\n🚨 活跃告警:\n"
            for alert in status['active_alerts']:
                severity_icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}[alert['severity']]
                report += f"  {severity_icon} {alert['message']} ({alert['timestamp']})\n"
        
        return report

# 全局健康监控器实例
_health_monitor = None

def get_health_monitor() -> SystemHealthMonitor:
    """获取全局健康监控器实例"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = SystemHealthMonitor()
    return _health_monitor

async def main():
    """主函数 - 健康监控测试"""
    monitor = get_health_monitor()
    
    print("🏥 启动系统健康监控测试...")
    
    # 启动监控
    monitor_task = await monitor.start_monitoring()
    
    try:
        # 运行一段时间收集数据
        await asyncio.sleep(60)
        
        # 获取健康状态
        status = monitor.get_health_status()
        print("\n📊 系统健康状态:")
        print(json.dumps(status, indent=2, default=str))
        
        # 生成健康报告
        report = await monitor.generate_health_report()
        print("\n📋 健康报告:")
        print(report)
        
    finally:
        await monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())