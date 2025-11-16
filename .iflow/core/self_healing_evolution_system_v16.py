#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ 自我修复和自适应进化系统 V16
================================

这是iFlow CLI的自我修复和自适应进化系统，实现：
- 自动错误检测和修复
- 性能监控和优化
- 系统自适应进化
- 反脆弱机制实现
- 持续学习和改进

你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）

作者: AI架构师团队
版本: 16.0.0
日期: 2025-11-16
"""

import os
import sys
import json
import asyncio
import logging
import time
import traceback
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

# 项目根路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SystemHealth:
    """系统健康状态"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    error_rate: float
    performance_score: float
    timestamp: datetime
    status: str  # "healthy", "warning", "critical"

@dataclass
class RepairAction:
    """修复动作"""
    action_id: str
    action_type: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    auto_repairable: bool
    repair_function: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)

class SelfHealingSystem:
    """自我修复系统"""
    
    def __init__(self):
        self.health_history = deque(maxlen=1000)
        self.repair_history = deque(maxlen=500)
        self.known_issues = {}
        self.repair_strategies = {}
        self.evolution_metrics = {
            "total_repairs": 0,
            "successful_repairs": 0,
            "prevention_count": 0,
            "evolution_score": 0.0
        }
        
        # 初始化修复策略
        self._initialize_repair_strategies()
        
    def _initialize_repair_strategies(self):
        """初始化修复策略"""
        self.repair_strategies = {
            "high_memory": {
                "action": "optimize_memory",
                "function": self._optimize_memory_usage,
                "threshold": 80.0
            },
            "high_cpu": {
                "action": "optimize_cpu",
                "function": self._optimize_cpu_usage,
                "threshold": 90.0
            },
            "disk_space_low": {
                "action": "cleanup_disk",
                "function": self._cleanup_disk_space,
                "threshold": 90.0
            },
            "error_spike": {
                "action": "restart_components",
                "function": self._restart_failing_components,
                "threshold": 0.1
            },
            "performance_degradation": {
                "action": "optimize_performance",
                "function": self._optimize_system_performance,
                "threshold": 0.7
            }
        }
    
    async def monitor_system_health(self) -> SystemHealth:
        """监控系统健康状态"""
        try:
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # 磁盘使用率
            disk = psutil.disk_usage(str(PROJECT_ROOT))
            disk_usage = disk.percent
            
            # 错误率（从日志分析）
            error_rate = await self._analyze_error_rate()
            
            # 性能评分
            performance_score = await self._calculate_performance_score()
            
            # 确定状态
            if cpu_usage > 90 or memory_usage > 90 or disk_usage > 95:
                status = "critical"
            elif cpu_usage > 70 or memory_usage > 70 or disk_usage > 80:
                status = "warning"
            else:
                status = "healthy"
            
            health = SystemHealth(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                error_rate=error_rate,
                performance_score=performance_score,
                timestamp=datetime.now(),
                status=status
            )
            
            self.health_history.append(health)
            return health
            
        except Exception as e:
            logger.error(f"健康监控失败: {e}")
            raise
    
    async def _analyze_error_rate(self) -> float:
        """分析错误率"""
        try:
            # 检查最近的错误日志
            log_files = [
                PROJECT_ROOT / ".iflow" / "logs" / "error.log",
                PROJECT_ROOT / "knowledge_base" / "logs" / "error.log"
            ]
            
            total_lines = 0
            error_lines = 0
            recent_time = datetime.now() - timedelta(hours=1)
            
            for log_file in log_files:
                if log_file.exists():
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            total_lines += 1
                            if "ERROR" in line or "CRITICAL" in line:
                                error_lines += 1
            
            return error_lines / max(total_lines, 1)
            
        except Exception:
            return 0.0
    
    async def _calculate_performance_score(self) -> float:
        """计算性能评分"""
        try:
            # 基于多个指标计算综合评分
            scores = []
            
            # CPU评分（越低越好）
            cpu_usage = psutil.cpu_percent()
            cpu_score = max(0, 1 - cpu_usage / 100)
            scores.append(cpu_score)
            
            # 内存评分
            memory = psutil.virtual_memory()
            memory_score = max(0, 1 - memory.percent / 100)
            scores.append(memory_score)
            
            # 响应时间评分（如果有历史数据）
            if len(self.health_history) > 0:
                recent_health = self.health_history[-1]
                response_score = recent_health.performance_score
                scores.append(response_score)
            
            return np.mean(scores) if scores else 0.8
            
        except Exception:
            return 0.5
    
    async def detect_issues(self, health: SystemHealth) -> List[RepairAction]:
        """检测系统问题"""
        issues = []
        
        # 检测内存问题
        if health.memory_usage > self.repair_strategies["high_memory"]["threshold"]:
            issues.append(RepairAction(
                action_id=f"mem_{int(time.time())}",
                action_type="memory_optimization",
                description=f"内存使用过高: {health.memory_usage:.1f}%",
                severity="high" if health.memory_usage > 90 else "medium",
                auto_repairable=True,
                repair_function=self.repair_strategies["high_memory"]["function"]
            ))
        
        # 检测CPU问题
        if health.cpu_usage > self.repair_strategies["high_cpu"]["threshold"]:
            issues.append(RepairAction(
                action_id=f"cpu_{int(time.time())}",
                action_type="cpu_optimization",
                description=f"CPU使用过高: {health.cpu_usage:.1f}%",
                severity="high" if health.cpu_usage > 95 else "medium",
                auto_repairable=True,
                repair_function=self.repair_strategies["high_cpu"]["function"]
            ))
        
        # 检测磁盘空间
        if health.disk_usage > self.repair_strategies["disk_space_low"]["threshold"]:
            issues.append(RepairAction(
                action_id=f"disk_{int(time.time())}",
                action_type="disk_cleanup",
                description=f"磁盘空间不足: {health.disk_usage:.1f}%",
                severity="critical",
                auto_repairable=True,
                repair_function=self.repair_strategies["disk_space_low"]["function"]
            ))
        
        # 检测错误率
        if health.error_rate > self.repair_strategies["error_spike"]["threshold"]:
            issues.append(RepairAction(
                action_id=f"err_{int(time.time())}",
                action_type="error_handling",
                description=f"错误率过高: {health.error_rate:.2%}",
                severity="high",
                auto_repairable=True,
                repair_function=self.repair_strategies["error_spike"]["function"]
            ))
        
        # 检测性能问题
        if health.performance_score < self.repair_strategies["performance_degradation"]["threshold"]:
            issues.append(RepairAction(
                action_id=f"perf_{int(time.time())}",
                action_type="performance_optimization",
                description=f"性能下降: {health.performance_score:.2f}",
                severity="medium",
                auto_repairable=True,
                repair_function=self.repair_strategies["performance_degradation"]["function"]
            ))
        
        return issues
    
    async def repair_issues(self, issues: List[RepairAction]) -> Dict[str, bool]:
        """修复问题"""
        results = {}
        
        for issue in issues:
            if issue.auto_repairable and issue.repair_function:
                try:
                    logger.info(f"开始修复: {issue.description}")
                    success = await issue.repair_function()
                    results[issue.action_id] = success
                    
                    if success:
                        logger.info(f"修复成功: {issue.action_id}")
                        self.evolution_metrics["successful_repairs"] += 1
                    else:
                        logger.warning(f"修复失败: {issue.action_id}")
                    
                    self.evolution_metrics["total_repairs"] += 1
                    self.repair_history.append({
                        "action_id": issue.action_id,
                        "success": success,
                        "timestamp": datetime.now()
                    })
                    
                except Exception as e:
                    logger.error(f"修复异常: {issue.action_id} - {e}")
                    results[issue.action_id] = False
            else:
                logger.warning(f"问题无法自动修复: {issue.description}")
                results[issue.action_id] = False
        
        return results
    
    async def _optimize_memory_usage(self) -> bool:
        """优化内存使用"""
        try:
            import gc
            
            # 强制垃圾回收
            gc.collect()
            
            # 清理临时文件
            temp_dirs = [
                PROJECT_ROOT / ".iflow" / "temp",
                PROJECT_ROOT / ".iflow" / "cache",
                PROJECT_ROOT / "temp"
            ]
            
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    for file in temp_dir.glob("*"):
                        if file.is_file():
                            try:
                                file.unlink()
                            except Exception:
                                pass
            
            # 优化向量索引缓存
            try:
                from improved_knowledge_base_manager_refactored import KnowledgeBaseManager
                kb = KnowledgeBaseManager()
                if hasattr(kb, 'optimize_memory'):
                    kb.optimize_memory()
            except Exception:
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"内存优化失败: {e}")
            return False
    
    async def _optimize_cpu_usage(self) -> bool:
        """优化CPU使用"""
        try:
            # 降低非关键进程优先级
            current_pid = os.getpid()
            p = psutil.Process(current_pid)
            
            # 设置为低优先级
            if hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS'):
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            
            # 优化并发任务
            try:
                # 减少并行度
                os.environ['OMP_NUM_THREADS'] = '2'
                os.environ['MKL_NUM_THREADS'] = '2'
            except Exception:
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"CPU优化失败: {e}")
            return False
    
    async def _cleanup_disk_space(self) -> bool:
        """清理磁盘空间"""
        try:
            # 清理日志文件
            log_dirs = [
                PROJECT_ROOT / ".iflow" / "logs",
                PROJECT_ROOT / "knowledge_base" / "logs"
            ]
            
            for log_dir in log_dirs:
                if log_dir.exists():
                    for log_file in log_dir.glob("*.log"):
                        if log_file.stat().st_size > 100 * 1024 * 1024:  # 大于100MB
                            # 截断日志文件
                            with open(log_file, 'r+', encoding='utf-8') as f:
                                f.seek(0, 2)  # 移到文件末尾
                                size = f.tell()
                                if size > 10 * 1024 * 1024:  # 保留最后10MB
                                    f.seek(size - 10 * 1024 * 1024)
                                    content = f.read()
                                    f.seek(0)
                                    f.truncate()
                                    f.write(content)
            
            # 清理旧的分析报告
            reports_dir = PROJECT_ROOT / "ARQ分析报告"
            if reports_dir.exists():
                cutoff_time = time.time() - 7 * 24 * 3600  # 7天前
                for report in reports_dir.glob("*.json"):
                    if report.stat().st_mtime < cutoff_time:
                        report.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"磁盘清理失败: {e}")
            return False
    
    async def _restart_failing_components(self) -> bool:
        """重启失败的组件"""
        try:
            # 重启知识库服务
            try:
                from knowledge_base_service import restart_kb_service
                restart_kb_service()
            except Exception:
                pass
            
            # 重新加载核心模块
            core_modules = [
                'arq_reasoning_engine_v16_quantum_evolution',
                'refrag_system_v5_quantum_compression',
                'hrrk_kernel_v3_enterprise'
            ]
            
            for module_name in core_modules:
                try:
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                except Exception:
                    pass
            
            return True
            
        except Exception as e:
            logger.error(f"组件重启失败: {e}")
            return False
    
    async def _optimize_system_performance(self) -> bool:
        """优化系统性能"""
        try:
            # 优化Python环境
            import gc
            gc.collect()
            
            # 预热关键组件
            try:
                from hrrk_kernel_v3_enterprise import HRRKKernelV3
                kernel = HRRKKernelV3()
                await kernel.warmup()
            except Exception:
                pass
            
            # 优化缓存
            try:
                from intelligent_cache import IntelligentCache
                cache = IntelligentCache()
                cache.optimize()
            except Exception:
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"性能优化失败: {e}")
            return False
    
    async def evolve_system(self) -> Dict[str, Any]:
        """系统进化"""
        evolution_report = {
            "timestamp": datetime.now().isoformat(),
            "evolution_score": 0.0,
            "improvements": [],
            "adaptive_changes": []
        }
        
        try:
            # 计算进化分数
            if self.evolution_metrics["total_repairs"] > 0:
                success_rate = self.evolution_metrics["successful_repairs"] / self.evolution_metrics["total_repairs"]
                self.evolution_metrics["evolution_score"] = min(1.0, success_rate * 1.2)
            
            evolution_report["evolution_score"] = self.evolution_metrics["evolution_score"]
            
            # 基于历史数据优化
            if len(self.health_history) > 10:
                recent_health = list(self.health_history)[-10:]
                avg_performance = np.mean([h.performance_score for h in recent_health])
                
                if avg_performance < 0.7:
                    # 自适应调整
                    evolution_report["improvements"].append("调整系统参数以提升性能")
                    await self._adaptive_tuning()
            
            # 预防性维护
            if len(self.repair_history) > 5:
                common_issues = self._analyze_common_issues()
                if common_issues:
                    evolution_report["adaptive_changes"].append(f"预防性修复: {common_issues}")
                    await self._preventive_maintenance(common_issues)
            
            return evolution_report
            
        except Exception as e:
            logger.error(f"系统进化失败: {e}")
            return evolution_report
    
    def _analyze_common_issues(self) -> List[str]:
        """分析常见问题"""
        issue_counts = defaultdict(int)
        
        for repair in self.repair_history:
            if not repair["success"]:
                action_id = repair["action_id"]
                issue_type = action_id.split("_")[0]
                issue_counts[issue_type] += 1
        
        # 返回最频繁的问题
        if issue_counts:
            most_common = max(issue_counts.items(), key=lambda x: x[1])
            if most_common[1] > 2:
                return [most_common[0]]
        
        return []
    
    async def _adaptive_tuning(self):
        """自适应调整"""
        try:
            # 动态调整配置
            config_updates = {}
            
            # 基于内存使用调整
            recent_memory = [h.memory_usage for h in list(self.health_history)[-5:]]
            if np.mean(recent_memory) > 70:
                config_updates["reduce_memory_usage"] = True
            
            # 基于CPU使用调整
            recent_cpu = [h.cpu_usage for h in list(self.health_history)[-5:]]
            if np.mean(recent_cpu) > 70:
                config_updates["reduce_cpu_usage"] = True
            
            # 应用配置更新
            if config_updates:
                await self._apply_config_updates(config_updates)
                
        except Exception as e:
            logger.error(f"自适应调整失败: {e}")
    
    async def _preventive_maintenance(self, issues: List[str]):
        """预防性维护"""
        try:
            for issue in issues:
                if issue == "mem":
                    await self._optimize_memory_usage()
                elif issue == "cpu":
                    await self._optimize_cpu_usage()
                elif issue == "disk":
                    await self._cleanup_disk_space()
                
                self.evolution_metrics["prevention_count"] += 1
                
        except Exception as e:
            logger.error(f"预防性维护失败: {e}")
    
    async def _apply_config_updates(self, updates: Dict[str, Any]):
        """应用配置更新"""
        try:
            config_file = PROJECT_ROOT / ".iflow" / "config" / "system_config.json"
            config_file.parent.mkdir(exist_ok=True)
            
            # 读取现有配置
            config = {}
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            
            # 更新配置
            config.update(updates)
            
            # 保存配置
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"配置更新失败: {e}")

class AdaptiveEvolutionEngine:
    """自适应进化引擎"""
    
    def __init__(self):
        self.healing_system = SelfHealingSystem()
        self.evolution_cycle = 300  # 5分钟
        self.running = False
        
    async def start_evolution(self):
        """启动进化循环"""
        self.running = True
        logger.info("自适应进化引擎启动")
        
        while self.running:
            try:
                # 监控系统健康
                health = await self.healing_system.monitor_system_health()
                
                # 检测问题
                issues = await self.healing_system.detect_issues(health)
                
                # 修复问题
                if issues:
                    logger.info(f"检测到 {len(issues)} 个问题，开始修复")
                    results = await self.healing_system.repair_issues(issues)
                    
                    success_count = sum(results.values())
                    logger.info(f"修复完成: {success_count}/{len(issues)} 成功")
                
                # 系统进化
                if len(self.healing_system.health_history) % 10 == 0:
                    evolution_report = await self.healing_system.evolve_system()
                    logger.info(f"进化分数: {evolution_report['evolution_score']:.2f}")
                
                # 等待下一个周期
                await asyncio.sleep(self.evolution_cycle)
                
            except Exception as e:
                logger.error(f"进化循环异常: {e}")
                await asyncio.sleep(60)  # 出错时等待1分钟
    
    def stop_evolution(self):
        """停止进化循环"""
        self.running = False
        logger.info("自适应进化引擎停止")

# 全局进化引擎实例
evolution_engine = AdaptiveEvolutionEngine()

async def start_self_healing_system():
    """启动自我修复系统"""
    logger.info("启动iFlow CLI自我修复系统V16")
    await evolution_engine.start_evolution()

def stop_self_healing_system():
    """停止自我修复系统"""
    evolution_engine.stop_evolution()

# 测试函数
async def test_self_healing():
    """测试自我修复系统"""
    print("🛡️ 测试自我修复系统...")
    
    # 监控健康
    health = await evolution_engine.healing_system.monitor_system_health()
    print(f"系统健康状态: {health.status}")
    
    # 检测问题
    issues = await evolution_engine.healing_system.detect_issues(health)
    print(f"检测到 {len(issues)} 个问题")
    
    # 修复问题
    if issues:
        results = await evolution_engine.healing_system.repair_issues(issues)
        success_count = sum(results.values())
        print(f"修复成功: {success_count}/{len(issues)}")
    
    print("✅ 自我修复系统测试完成")

# 添加SelfHealingEvolutionSystemV16类以兼容工作流
class SelfHealingEvolutionSystemV16(AdaptiveEvolutionEngine):
    """自我修复进化系统V16 - 兼容性包装器"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or {}
    
    async def check_and_heal(self, result: Dict):
        """检查并修复结果"""
        try:
            # 监控系统健康
            health = await self.healing_system.monitor_system_health()
            
            # 检测问题
            issues = await self.healing_system.detect_issues(health)
            
            # 修复问题
            if issues:
                await self.healing_system.repair_issues(issues)
            
            return {"healing_status": "completed", "issues_found": len(issues)}
        except Exception as e:
            return {"healing_status": "failed", "error": str(e)}
    
    async def heal_error(self, error: Exception):
        """修复错误"""
        try:
            # 记录错误
            logger.error(f"自我修复错误: {error}")
            
            # 尝试修复
            health = await self.healing_system.monitor_system_health()
            issues = await self.healing_system.detect_issues(health)
            
            if issues:
                await self.healing_system.repair_issues(issues)
            
            return True
        except Exception:
            return False
    
    async def cleanup(self):
        """清理资源"""
        self.stop_evolution()

if __name__ == "__main__":
    asyncio.run(test_self_healing())