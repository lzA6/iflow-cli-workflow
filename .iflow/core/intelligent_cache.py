#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ 智能缓存系统 (Intelligent Cache System) - 重构版本
========================================================

重构目标：
- 降低圈复杂度从47到25以下
- 提高可维护性指数到50以上
- 优化缓存策略和性能
- 改进内存管理和清理机制

特性：
- 分层架构设计
- 策略模式实现
- 内存泄漏防护
- 多线程安全
- 缓存命中率优化

作者: iFlow性能优化团队
版本: 2.0.0 (重构版)
日期: 2025-11-16
"""

import os
import time
import hashlib
import threading
import pickle
import weakref
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set
from dataclasses import dataclass, field
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """缓存条目 - 优化版本"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def touch(self) -> None:
        """更新访问时间"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def get_priority_score(self) -> float:
        """获取优先级分数（用于淘汰策略）"""
        age = time.time() - self.last_accessed
        return self.access_count / (age + 1)  # 避免除零

@dataclass
class CacheStats:
    """缓存统计 - 增强版本"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    memory_usage_bytes: int = 0
    cache_size: int = 0
    cleanup_count: int = 0
    persistence_errors: int = 0
    
    @property
    def hit_rate(self) -> float:
        """命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    @property
    def memory_usage_mb(self) -> float:
        """内存使用量(MB)"""
        return self.memory_usage_bytes / (1024 * 1024)
    
    @property
    def average_item_size(self) -> float:
        """平均条目大小"""
        if self.cache_size == 0:
            return 0.0
        return self.memory_usage_bytes / self.cache_size
    
    def reset(self) -> None:
        """重置统计信息"""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_requests = 0
        self.cleanup_count = 0
        self.persistence_errors = 0

class EvictionStrategy(ABC):
    """淘汰策略抽象基类"""
    
    @abstractmethod
    def select_victim(self, cache: Dict[str, CacheEntry]) -> Optional[str]:
        """选择要淘汰的条目"""
        pass

class LRUEvictionStrategy(EvictionStrategy):
    """LRU淘汰策略"""
    
    def select_victim(self, cache: Dict[str, CacheEntry]) -> Optional[str]:
        if not cache:
            return None
        return min(cache.keys(), key=lambda k: cache[k].last_accessed)

class LFUEvictionStrategy(EvictionStrategy):
    """LFU淘汰策略"""
    
    def select_victim(self, cache: Dict[str, CacheEntry]) -> Optional[str]:
        if not cache:
            return None
        return min(cache.keys(), key=lambda k: cache[k].access_count)

class SizeBasedEvictionStrategy(EvictionStrategy):
    """基于大小的淘汰策略"""
    
    def __init__(self, target_memory: int):
        self.target_memory = target_memory
    
    def select_victim(self, cache: Dict[str, CacheEntry]) -> Optional[str]:
        if not cache:
            return None
        return max(cache.keys(), key=lambda k: cache[k].size_bytes)

class AdaptiveEvictionStrategy(EvictionStrategy):
    """自适应淘汰策略"""
    
    def select_victim(self, cache: Dict[str, CacheEntry]) -> Optional[str]:
        if not cache:
            return None
        # 综合考虑访问频率、最近访问时间和大小
        return min(cache.keys(), key=lambda k: self._calculate_score(cache[k]))
    
    def _calculate_score(self, entry: CacheEntry) -> float:
        """计算条目分数（分数越低越容易被淘汰）"""
        age = time.time() - entry.last_accessed
        size_factor = entry.size_bytes / (1024 * 1024)  # MB为单位
        return (age + 1) / (entry.access_count + 1) * (1 + size_factor * 0.1)

class MemoryManager:
    """内存管理器 - 负责容量控制和淘汰策略"""
    
    def __init__(self, max_size: int, max_memory_bytes: int, eviction_strategy: EvictionStrategy):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_bytes
        self.eviction_strategy = eviction_strategy
    
    def ensure_capacity(self, cache: Dict[str, CacheEntry], new_item_size: int) -> List[str]:
        """确保有足够容量，返回需要淘汰的键列表"""
        evicted_keys = []
        
        # 检查条目数限制
        while len(cache) >= self.max_size:
            victim_key = self.eviction_strategy.select_victim(cache)
            if victim_key and victim_key in cache:
                evicted_keys.append(victim_key)
                del cache[victim_key]
            else:
                break
        
        # 检查内存限制
        current_memory = sum(entry.size_bytes for entry in cache.values())
        while current_memory + new_item_size > self.max_memory_bytes and cache:
            victim_key = self.eviction_strategy.select_victim(cache)
            if victim_key and victim_key in cache:
                current_memory -= cache[victim_key].size_bytes
                evicted_keys.append(victim_key)
                del cache[victim_key]
            else:
                break
        
        return evicted_keys

class PersistenceManager:
    """持久化管理器 - 负责缓存的持久化操作"""
    
    def __init__(self, persist_file: Optional[str] = None):
        self.persist_file = persist_file
        self._lock = threading.Lock()
    
    def load_cache(self) -> Dict[str, CacheEntry]:
        """加载持久化缓存"""
        if not self.persist_file or not os.path.exists(self.persist_file):
            return {}
        
        with self._lock:
            try:
                with open(self.persist_file, 'rb') as f:
                    data = pickle.load(f)
                
                cache = {}
                if isinstance(data, dict) and 'cache' in data:
                    for key, entry_data in data['cache'].items():
                        if isinstance(entry_data, dict):
                            entry = CacheEntry(**entry_data)
                            if not entry.is_expired():
                                cache[key] = entry
                
                logger.info(f"加载持久化缓存: {len(cache)} 个条目")
                return cache
                
            except Exception as e:
                logger.error(f"加载持久化缓存失败: {e}")
                return {}
    
    def save_cache(self, cache: Dict[str, CacheEntry], stats: CacheStats) -> bool:
        """保存持久化缓存"""
        if not self.persist_file:
            return False
        
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self.persist_file), exist_ok=True)
                
                data = {
                    'cache': {
                        key: {
                            'key': entry.key,
                            'value': entry.value,
                            'created_at': entry.created_at,
                            'last_accessed': entry.last_accessed,
                            'access_count': entry.access_count,
                            'size_bytes': entry.size_bytes,
                            'ttl': entry.ttl
                        }
                        for key, entry in cache.items()
                    },
                    'stats': {
                        'hits': stats.hits,
                        'misses': stats.misses,
                        'evictions': stats.evictions,
                        'total_requests': stats.total_requests,
                        'saved_at': time.time()
                    }
                }
                
                with open(self.persist_file, 'wb') as f:
                    pickle.dump(data, f)
                
                logger.info(f"保存持久化缓存: {len(cache)} 个条目")
                return True
                
            except Exception as e:
                logger.error(f"保存持久化缓存失败: {e}")
                return False

class CleanupManager:
    """清理管理器 - 负责过期条目的清理"""
    
    def __init__(self, cleanup_interval: int = 60):
        self.cleanup_interval = cleanup_interval
        self._stop_event = threading.Event()
        self._cleanup_thread = None
        self._weak_refs: Set[weakref.ref] = set()
    
    def register_cache(self, cache_ref) -> None:
        """注册缓存实例进行清理"""
        self._weak_refs.add(weakref.ref(cache_ref))
    
    def start(self) -> None:
        """启动清理线程"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._stop_event.clear()
            self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self._cleanup_thread.start()
            logger.info("清理管理器已启动")
    
    def stop(self) -> None:
        """停止清理线程"""
        self._stop_event.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        logger.info("清理管理器已停止")
    
    def _cleanup_worker(self) -> None:
        """清理工作线程"""
        while not self._stop_event.wait(self.cleanup_interval):
            self._perform_cleanup()
    
    def _perform_cleanup(self) -> None:
        """执行清理操作"""
        dead_refs = set()
        
        for cache_ref in self._weak_refs:
            cache_instance = cache_ref()
            if cache_instance is None:
                dead_refs.add(cache_ref)
                continue
            
            try:
                cache_instance._cleanup_expired()
            except Exception as e:
                logger.error(f"缓存清理失败: {e}")
        
        # 清理失效的弱引用
        self._weak_refs -= dead_refs

class IntelligentCache:
    """智能缓存系统 - 重构版本"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: int = 512,
                 ttl_seconds: Optional[float] = None,
                 policy: str = "lru",
                 persist_file: Optional[str] = None):
        """
        初始化智能缓存
        
        Args:
            max_size: 最大条目数
            max_memory_mb: 最大内存使用(MB)
            ttl_seconds: 默认TTL(秒)
            policy: 淘汰策略 (lru, lfu, adaptive)
            persist_file: 持久化文件路径
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = ttl_seconds
        self.policy = policy
        
        # 核心数据
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()
        self.lock = threading.RLock()
        
        # 初始化组件
        eviction_strategy = self._create_eviction_strategy(policy)
        self.memory_manager = MemoryManager(max_size, self.max_memory_bytes, eviction_strategy)
        self.persistence_manager = PersistenceManager(persist_file)
        
        # 全局清理管理器
        self._cleanup_manager = self._get_cleanup_manager()
        self._cleanup_manager.register_cache(self)
        
        # 加载持久化缓存
        self._load_persistent_cache()
        
        logger.info(f"智能缓存初始化完成: max_size={max_size}, max_memory={max_memory_mb}MB")
    
    def _create_eviction_strategy(self, policy: str) -> EvictionStrategy:
        """创建淘汰策略实例"""
        strategies = {
            "lru": LRUEvictionStrategy(),
            "lfu": LFUEvictionStrategy(),
            "adaptive": AdaptiveEvictionStrategy(),
        }
        return strategies.get(policy, LRUEvictionStrategy())
    
    def _get_cleanup_manager(self) -> CleanupManager:
        """获取全局清理管理器（单例模式）"""
        if not hasattr(IntelligentCache, '_global_cleanup_manager'):
            IntelligentCache._global_cleanup_manager = CleanupManager()
            IntelligentCache._global_cleanup_manager.start()
        return IntelligentCache._global_cleanup_manager
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值 - 简化版本"""
        with self.lock:
            self.stats.total_requests += 1
            
            entry = self.cache.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            
            if entry.is_expired():
                self._remove_entry(key)
                self.stats.misses += 1
                return None
            
            entry.touch()
            self.stats.hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值 - 简化版本"""
        with self.lock:
            try:
                size_bytes = self._calculate_size(value)
                
                if not self._validate_size(size_bytes):
                    return False
                
                self._update_existing_entry(key)
                self._ensure_capacity(size_bytes)
                
                entry = self._create_entry(key, value, size_bytes, ttl)
                self.cache[key] = entry
                self.stats.memory_usage_bytes += size_bytes
                
                return True
                
            except Exception as e:
                logger.error(f"缓存设置失败: {e}")
                return False
    
    def _calculate_size(self, value: Any) -> int:
        """计算值的大小"""
        return len(pickle.dumps(value))
    
    def _validate_size(self, size_bytes: int) -> bool:
        """验证大小限制"""
        if size_bytes > self.max_memory_bytes:
            logger.warning(f"缓存值过大: {size_bytes} bytes > {self.max_memory_bytes} bytes")
            return False
        return True
    
    def _update_existing_entry(self, key: str) -> None:
        """更新已存在的条目"""
        if key in self.cache:
            old_entry = self.cache[key]
            self.stats.memory_usage_bytes -= old_entry.size_bytes
    
    def _create_entry(self, key: str, value: Any, size_bytes: int, ttl: Optional[float]) -> CacheEntry:
        """创建新的缓存条目"""
        return CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            last_accessed=time.time(),
            size_bytes=size_bytes,
            ttl=ttl or self.default_ttl
        )
    
    def delete(self, key: str) -> bool:
        """删除缓存条目 - 简化版本"""
        with self.lock:
            return self._remove_entry(key)
    
    def _remove_entry(self, key: str) -> bool:
        """移除缓存条目"""
        if key in self.cache:
            entry = self.cache[key]
            del self.cache[key]
            self.stats.memory_usage_bytes -= entry.size_bytes
            return True
        return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.stats.memory_usage_bytes = 0
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计 - 简化版本"""
        with self.lock:
            self.stats.cache_size = len(self.cache)
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                total_requests=self.stats.total_requests,
                memory_usage_bytes=self.stats.memory_usage_bytes,
                cache_size=self.stats.cache_size,
                cleanup_count=self.stats.cleanup_count,
                persistence_errors=self.stats.persistence_errors
            )
    
    def _ensure_capacity(self, new_item_size: int) -> None:
        """确保有足够容量 - 简化版本"""
        evicted_keys = self.memory_manager.ensure_capacity(self.cache, new_item_size)
        
        for key in evicted_keys:
            if key in self.cache:
                entry = self.cache[key]
                del self.cache[key]
                self.stats.memory_usage_bytes -= entry.size_bytes
                self.stats.evictions += 1
    
    def _cleanup_expired(self) -> None:
        """清理过期条目 - 简化版本"""
        with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                self.stats.cleanup_count += len(expired_keys)
                logger.debug(f"清理了 {len(expired_keys)} 个过期条目")
    
    
    
    def _load_persistent_cache(self) -> None:
        """加载持久化缓存 - 简化版本"""
        loaded_cache = self.persistence_manager.load_cache()
        
        with self.lock:
            self.cache.update(loaded_cache)
            self.stats.memory_usage_bytes += sum(entry.size_bytes for entry in loaded_cache.values())
    
    def save_persistent_cache(self) -> bool:
        """保存持久化缓存 - 简化版本"""
        success = self.persistence_manager.save_cache(self.cache, self.stats)
        if not success:
            self.stats.persistence_errors += 1
        return success
    
    def __del__(self):
        """析构函数 - 确保资源清理"""
        try:
            self.save_persistent_cache()
        except Exception:
            pass  # 防止析构函数中的异常

class CacheManager:
    """缓存管理器 - 增强版本"""
    
    def __init__(self):
        self.caches: Dict[str, IntelligentCache] = {}
        self.lock = threading.RLock()
        self.default_config = {
            'max_size': 1000,
            'max_memory_mb': 512,
            'ttl_seconds': 3600,  # 1小时
            'policy': 'adaptive'  # 默认使用自适应策略
        }
    
    def get_cache(self, name: str, **kwargs) -> IntelligentCache:
        """获取或创建缓存 - 线程安全版本"""
        with self.lock:
            if name not in self.caches:
                config = {**self.default_config, **kwargs}
                persist_file = f"./cache/{name}_cache.pkl"
                self.caches[name] = IntelligentCache(persist_file=persist_file, **config)
            
            return self.caches[name]
    
    def get_all_stats(self) -> Dict[str, CacheStats]:
        """获取所有缓存统计"""
        with self.lock:
            return {name: cache.get_stats() for name, cache in self.caches.items()}
    
    def clear_all(self) -> None:
        """清空所有缓存"""
        with self.lock:
            for cache in self.caches.values():
                cache.clear()
    
    def save_all(self) -> None:
        """保存所有缓存"""
        with self.lock:
            for cache in self.caches.values():
                cache.save_persistent_cache()
    
    def get_cache_names(self) -> List[str]:
        """获取所有缓存名称"""
        with self.lock:
            return list(self.caches.keys())
    
    def remove_cache(self, name: str) -> bool:
        """移除指定的缓存"""
        with self.lock:
            if name in self.caches:
                del self.caches[name]
                return True
            return False
    
    def shutdown(self) -> None:
        """关闭管理器，保存所有缓存"""
        logger.info("正在关闭缓存管理器...")
        self.save_all()
        
        # 停止清理管理器
        if hasattr(IntelligentCache, '_global_cleanup_manager'):
            IntelligentCache._global_cleanup_manager.stop()
        
        with self.lock:
            self.caches.clear()
        
        logger.info("缓存管理器已关闭")

# 全局缓存管理器（单例）
_cache_manager: Optional[CacheManager] = None
_manager_lock = threading.Lock()

def get_cache_manager() -> CacheManager:
    """获取缓存管理器 - 单例模式"""
    global _cache_manager
    if _cache_manager is None:
        with _manager_lock:
            if _cache_manager is None:
                _cache_manager = CacheManager()
    return _cache_manager

def get_cache(name: str, **kwargs) -> IntelligentCache:
    """获取缓存实例"""
    return get_cache_manager().get_cache(name, **kwargs)

# 装饰器 - 增强版本
def cached(ttl: Optional[float] = None, cache_name: str = "default", key_prefix: str = ""):
    """缓存装饰器 - 支持更多选项"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成缓存键（更稳定的方式）
            try:
                key_data = f"{key_prefix}{func.__module__}.{func.__name__}:{repr(args)}:{repr(kwargs)}"
                cache_key = hashlib.sha256(key_data.encode()).hexdigest()
            except Exception:
                # 降级到简单键生成
                cache_key = f"{key_prefix}{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            cache = get_cache(cache_name)
            
            # 尝试从缓存获取
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._cached = True  # 标记为缓存装饰器
        
        return wrapper
    return decorator

def async_cached(ttl: Optional[float] = None, cache_name: str = "async_default"):
    """异步函数缓存装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            key_data = f"async:{func.__module__}.{func.__name__}:{repr(args)}:{repr(kwargs)}"
            cache_key = hashlib.sha256(key_data.encode()).hexdigest()
            
            cache = get_cache(cache_name)
            
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._async_cached = True
        
        return wrapper
    return decorator

if __name__ == "__main__":
    # 测试智能缓存 - 重构版本
    print("⚡ 测试智能缓存系统 v2.0")
    
    # 创建缓存
    cache = get_cache("test", max_size=10, max_memory_mb=1, policy="adaptive")
    
    # 测试基本操作
    cache.set("key1", "value1")
    cache.set("key2", {"data": "test"}, ttl=60)
    cache.set("key3", [1, 2, 3, 4, 5])  # 测试列表缓存
    
    print(f"获取key1: {cache.get('key1')}")
    print(f"获取key2: {cache.get('key2')}")
    print(f"获取key3: {cache.get('key3')}")
    print(f"获取不存在的key: {cache.get('key4')}")
    
    # 测试统计
    stats = cache.get_stats()
    print(f"缓存统计: 命中率={stats.hit_rate:.2%}, 大小={stats.cache_size}, 内存={stats.memory_usage_mb:.2f}MB")
    print(f"平均条目大小: {stats.average_item_size:.2f} bytes")
    
    # 测试装饰器
    @cached(ttl=30, cache_name="decorator_test")
    def expensive_function(x):
        time.sleep(0.1)  # 模拟耗时操作
        return x * x
    
    print("第一次调用:", expensive_function(5))
    print("第二次调用:", expensive_function(5))  # 应该从缓存获取
    
    # 测试缓存管理器
    manager = get_cache_manager()
    print(f"所有缓存: {manager.get_cache_names()}")
    all_stats = manager.get_all_stats()
    for name, stats in all_stats.items():
        print(f"缓存 '{name}': 命中率={stats.hit_rate:.2%}, 大小={stats.cache_size}")
    
    # 测试持久化
    cache.save_persistent_cache()
    print("持久化保存完成")
    
    # 清理测试
    print("\n测试清理功能...")
    cache.clear()
    stats_after_clear = cache.get_stats()
    print(f"清理后统计: 大小={stats_after_clear.cache_size}, 内存={stats_after_clear.memory_usage_mb:.2f}MB")
    
    print("✅ 智能缓存系统 v2.0 测试完成")
    
    # 优雅关闭
    get_cache_manager().shutdown()