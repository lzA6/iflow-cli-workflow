#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 内存优化工具 (Memory Optimization Tools)
=============================================

提供内存使用优化和管理功能：
- 内存使用监控
- 自动垃圾回收
- 内存泄漏检测
- 大对象优化
- 内存池管理

特性：
- 实时内存监控
- 智能垃圾回收
- 内存使用分析
- 性能优化建议

作者: iFlow性能优化团队
版本: 1.0.0
日期: 2025-11-16
"""

import gc
import sys
import time
import threading
import psutil
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import weakref
import logging

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """内存统计信息"""
    total_mb: float = 0.0
    used_mb: float = 0.0
    available_mb: float = 0.0
    percent_used: float = 0.0
    process_mb: float = 0.0
    gc_counts: Dict[int, int] = field(default_factory=dict)
    object_count: int = 0
    large_objects: List[str] = field(default_factory=list)

@dataclass
class MemoryLeakInfo:
    """内存泄漏信息"""
    object_type: str
    count: int
    size_mb: float
    growth_rate: float
    suspicious: bool = False

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, check_interval: float = 30.0, alert_threshold: float = 80.0):
        """
        初始化内存监控器
        
        Args:
            check_interval: 检查间隔(秒)
            alert_threshold: 内存使用警告阈值(%)
        """
        self.check_interval = check_interval
        self.alert_threshold = alert_threshold
        self.process = psutil.Process()
        
        # 监控历史
        self.history: List[MemoryStats] = []
        self.max_history = 100
        
        # 监控线程
        self.monitoring = False
        self.monitor_thread = None
        
        # 回调函数
        self.alert_callbacks: List[Callable[[MemoryStats], None]] = []
        
        # 对象跟踪
        self.object_tracker = ObjectTracker()
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        self.monitor_thread.start()
        logger.info("内存监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("内存监控已停止")
    
    def add_alert_callback(self, callback: Callable[[MemoryStats], None]):
        """添加警告回调"""
        self.alert_callbacks.append(callback)
    
    def get_current_stats(self) -> MemoryStats:
        """获取当前内存统计"""
        # 系统内存
        memory = psutil.virtual_memory()
        
        # 进程内存
        process_memory = self.process.memory_info()
        
        # GC统计
        gc_stats = gc.get_count()
        gc_counts = {i: gc_stats[i] for i in range(len(gc_stats))}
        
        # 对象计数
        object_count = len(gc.get_objects())
        
        # 大对象检测
        large_objects = self._detect_large_objects()
        
        return MemoryStats(
            total_mb=memory.total / (1024 * 1024),
            used_mb=memory.used / (1024 * 1024),
            available_mb=memory.available / (1024 * 1024),
            percent_used=memory.percent,
            process_mb=process_memory.rss / (1024 * 1024),
            gc_counts=gc_counts,
            object_count=object_count,
            large_objects=large_objects
        )
    
    def _detect_large_objects(self, threshold_mb: float = 1.0) -> List[str]:
        """检测大对象"""
        large_objects = []
        threshold_bytes = threshold_mb * 1024 * 1024
        
        try:
            all_objects = gc.get_objects()
            for obj in all_objects[:1000]:  # 限制检查数量以避免性能问题
                try:
                    size = sys.getsizeof(obj)
                    if size > threshold_bytes:
                        obj_type = type(obj).__name__
                        obj_id = id(obj)
                        large_objects.append(f"{obj_type}:{obj_id}:{size/1024/1024:.2f}MB")
                except:
                    continue
        except Exception as e:
            logger.debug(f"大对象检测失败: {e}")
        
        return large_objects[:10]  # 返回前10个最大的对象
    
    def _monitor_worker(self):
        """监控工作线程"""
        while self.monitoring:
            try:
                stats = self.get_current_stats()
                
                # 添加到历史
                self.history.append(stats)
                if len(self.history) > self.max_history:
                    self.history.pop(0)
                
                # 检查警告阈值
                if stats.percent_used > self.alert_threshold:
                    self._trigger_alert(stats)
                
                # 自动垃圾回收
                if stats.percent_used > 90:
                    self._auto_gc()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"内存监控错误: {e}")
                time.sleep(self.check_interval)
    
    def _trigger_alert(self, stats: MemoryStats):
        """触发内存警告"""
        logger.warning(f"内存使用过高: {stats.percent_used:.1f}%")
        
        for callback in self.alert_callbacks:
            try:
                callback(stats)
            except Exception as e:
                logger.error(f"内存警告回调失败: {e}")
    
    def _auto_gc(self):
        """自动垃圾回收"""
        logger.info("执行自动垃圾回收")
        
        # 执行多代垃圾回收
        collected = gc.collect()
        logger.info(f"垃圾回收完成: 回收了 {collected} 个对象")
    
    def detect_memory_leaks(self, window_minutes: int = 10) -> List[MemoryLeakInfo]:
        """检测内存泄漏"""
        if len(self.history) < 2:
            return []
        
        # 计算时间窗口
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        # 过滤历史数据
        recent_stats = [
            stat for stat in self.history
            if current_time - (len(self.history) - list(reversed(self.history)).index(stat)) * self.check_interval >= window_start
        ]
        
        if len(recent_stats) < 2:
            return []
        
        # 分析内存增长
        leaks = []
        
        # 进程内存增长
        first_process = recent_stats[0].process_mb
        last_process = recent_stats[-1].process_mb
        process_growth = (last_process - first_process) / first_process if first_process > 0 else 0
        
        if process_growth > 0.5:  # 50%增长
            leaks.append(MemoryLeakInfo(
                object_type="ProcessMemory",
                count=1,
                size_mb=last_process - first_process,
                growth_rate=process_growth,
                suspicious=process_growth > 1.0
            ))
        
        # 对象数量增长
        first_objects = recent_stats[0].object_count
        last_objects = recent_stats[-1].object_count
        object_growth = (last_objects - first_objects) / first_objects if first_objects > 0 else 0
        
        if object_growth > 0.3:  # 30%增长
            leaks.append(MemoryLeakInfo(
                object_type="ObjectCount",
                count=last_objects - first_objects,
                size_mb=0,  # 无法精确计算
                growth_rate=object_growth,
                suspicious=object_growth > 0.5
            ))
        
        return leaks

class ObjectTracker:
    """对象跟踪器"""
    
    def __init__(self):
        self.tracked_objects: Dict[int, weakref.ref] = {}
        self.object_types: Dict[str, int] = {}
    
    def track_object(self, obj: Any, name: Optional[str] = None):
        """跟踪对象"""
        obj_id = id(obj)
        obj_type = type(obj).__name__
        
        # 使用弱引用避免影响垃圾回收
        def cleanup(ref):
            if obj_id in self.tracked_objects:
                del self.tracked_objects[obj_id]
                self.object_types[obj_type] = self.object_types.get(obj_type, 0) - 1
                if self.object_types[obj_type] <= 0:
                    del self.object_types[obj_type]
        
        self.tracked_objects[obj_id] = weakref.ref(obj, cleanup)
        self.object_types[obj_type] = self.object_types.get(obj_type, 0) + 1
    
    def get_tracked_counts(self) -> Dict[str, int]:
        """获取跟踪的对象计数"""
        return self.object_types.copy()
    
    def cleanup_dead_references(self):
        """清理死引用"""
        dead_refs = [obj_id for obj_id, ref in self.tracked_objects.items() if ref() is None]
        for obj_id in dead_refs:
            del self.tracked_objects[obj_id]

class MemoryPool:
    """内存池"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.pool: List[Any] = []
        self.lock = threading.Lock()
    
    def get_object(self, object_type: type, *args, **kwargs) -> Any:
        """从池中获取对象"""
        with self.lock:
            for obj in self.pool:
                if isinstance(obj, object_type):
                    self.pool.remove(obj)
                    return obj
            
            # 池中没有，创建新对象
            return object_type(*args, **kwargs)
    
    def return_object(self, obj: Any):
        """将对象返回池中"""
        with self.lock:
            if len(self.pool) < self.max_size:
                # 重置对象状态
                if hasattr(obj, 'reset'):
                    obj.reset()
                self.pool.append(obj)
    
    def clear_pool(self):
        """清空池"""
        with self.lock:
            self.pool.clear()

class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self):
        self.monitor = MemoryMonitor()
        self.object_tracker = ObjectTracker()
        self.pools: Dict[str, MemoryPool] = {}
        
        # 优化策略
        self.optimization_strategies = {
            'gc_tuning': self._tune_gc,
            'object_pooling': self._optimize_object_pooling,
            'large_object_handling': self._optimize_large_objects,
            'memory_pressure_handling': self._handle_memory_pressure
        }
    
    def start_optimization(self):
        """开始内存优化"""
        self.monitor.start_monitoring()
        self.monitor.add_alert_callback(self._on_memory_alert)
        logger.info("内存优化已启动")
    
    def stop_optimization(self):
        """停止内存优化"""
        self.monitor.stop_monitoring()
        logger.info("内存优化已停止")
    
    def get_memory_stats(self) -> MemoryStats:
        """获取内存统计"""
        return self.monitor.get_current_stats()
    
    def detect_leaks(self) -> List[MemoryLeakInfo]:
        """检测内存泄漏"""
        return self.monitor.detect_memory_leaks()
    
    def optimize_memory(self, strategies: Optional[List[str]] = None):
        """执行内存优化"""
        if strategies is None:
            strategies = list(self.optimization_strategies.keys())
        
        for strategy in strategies:
            if strategy in self.optimization_strategies:
                try:
                    self.optimization_strategies[strategy]()
                    logger.info(f"执行内存优化策略: {strategy}")
                except Exception as e:
                    logger.error(f"内存优化策略失败 {strategy}: {e}")
    
    def _on_memory_alert(self, stats: MemoryStats):
        """内存警告回调"""
        logger.warning(f"内存警告: {stats.percent_used:.1f}% 使用率")
        
        # 自动优化
        if stats.percent_used > 85:
            self.optimize_memory(['gc_tuning', 'memory_pressure_handling'])
    
    def _tune_gc(self):
        """调优垃圾回收"""
        # 设置垃圾回收阈值
        gc.set_threshold(700, 10, 10)
        
        # 执行垃圾回收
        collected = gc.collect()
        logger.info(f"垃圾回收调优完成: 回收 {collected} 个对象")
    
    def _optimize_object_pooling(self):
        """优化对象池"""
        # 清理死引用
        self.object_tracker.cleanup_dead_references()
        
        # 清理过大的池
        for pool in self.pools.values():
            if len(pool.pool) > pool.max_size * 0.8:
                pool.clear_pool()
    
    def _optimize_large_objects(self):
        """优化大对象"""
        stats = self.monitor.get_current_stats()
        
        # 处理大对象
        for obj_info in stats.large_objects:
            try:
                obj_type, obj_id, size = obj_info.split(':')
                logger.warning(f"发现大对象: {obj_type} {size}MB")
            except:
                continue
    
    def _handle_memory_pressure(self):
        """处理内存压力"""
        stats = self.monitor.get_current_stats()
        
        if stats.percent_used > 90:
            # 紧急内存清理
            logger.warning("执行紧急内存清理")
            
            # 强制垃圾回收
            gc.collect()
            
            # 清理所有对象池
            for pool in self.pools.values():
                pool.clear_pool()
            
            # 清理跟踪器
            self.object_tracker.cleanup_dead_references()

# 全局内存优化器
_global_optimizer: Optional[MemoryOptimizer] = None

def get_memory_optimizer() -> MemoryOptimizer:
    """获取全局内存优化器"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = MemoryOptimizer()
    return _global_optimizer

def start_memory_optimization():
    """启动内存优化"""
    optimizer = get_memory_optimizer()
    optimizer.start_optimization()
    return optimizer

def stop_memory_optimization():
    """停止内存优化"""
    optimizer = get_memory_optimizer()
    optimizer.stop_optimization()

# 装饰器
def memory_efficient(max_size_mb: float = 10.0):
    """内存效率装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 检查内存使用
            optimizer = get_memory_optimizer()
            stats = optimizer.get_memory_stats()
            
            if stats.process_mb > max_size_mb:
                optimizer.optimize_memory()
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # 测试内存优化工具
    print("🧠 测试内存优化工具")
    
    # 启动内存优化
    optimizer = start_memory_optimization()
    
    # 获取内存统计
    stats = optimizer.get_memory_stats()
    print(f"当前内存使用: {stats.process_mb:.2f}MB ({stats.percent_used:.1f}%)")
    print(f"对象数量: {stats.object_count}")
    print(f"GC统计: {stats.gc_counts}")
    
    # 测试对象跟踪
    test_data = []
    for i in range(1000):
        data = {"id": i, "data": "x" * 100}
        optimizer.object_tracker.track_object(data)
        test_data.append(data)
    
    # 检查跟踪结果
    counts = optimizer.object_tracker.get_tracked_counts()
    print(f"跟踪的对象类型: {counts}")
    
    # 模拟内存压力
    large_data = "x" * (10 * 1024 * 1024)  # 10MB
    print(f"创建大对象后内存: {optimizer.get_memory_stats().process_mb:.2f}MB")
    
    # 执行内存优化
    optimizer.optimize_memory()
    
    # 清理
    del large_data
    del test_data
    
    print("✅ 内存优化工具测试完成")