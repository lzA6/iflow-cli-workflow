#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iFlow CLI 智能缓存集成
为所有iFlow命令提供透明的缓存支持
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, Optional

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from auto_cache_manager import get_auto_manager
    AUTO_CACHE_AVAILABLE = True
except ImportError:
    AUTO_CACHE_AVAILABLE = False

class SmartCache:
    """智能缓存系统"""
    
    def __init__(self):
        self.cache_manager = get_auto_manager() if AUTO_CACHE_AVAILABLE else None
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "stored": 0
        }
    
    def get_cache_key(self, prefix: str, data: Any) -> str:
        """生成缓存键"""
        # 将数据序列化为字符串
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        # 生成哈希
        hash_value = hashlib.md5(data_str.encode()).hexdigest()
        return f"{prefix}_{hash_value[:16]}"
    
    def get(self, prefix: str, data: Any, ttl_hours: int = 1) -> Optional[Any]:
        """获取缓存"""
        if not self.cache_manager:
            return None
        
        cache_key = self.get_cache_key(prefix, data)
        cached_data = self.cache_manager.get_temp(cache_key)
        
        if cached_data:
            self.cache_stats["hits"] += 1
            return cached_data.get("data")
        else:
            self.cache_stats["misses"] += 1
            return None
    
    def set(self, prefix: str, data: Any, result: Any, ttl_hours: int = 1):
        """设置缓存"""
        if not self.cache_manager:
            return
        
        cache_key = self.get_cache_key(prefix, data)
        cache_data = {
            "data": result,
            "timestamp": datetime.now().isoformat(),
            "ttl_hours": ttl_hours
        }
        
        self.cache_manager.store_temp(cache_data, cache_key, ttl_hours)
        self.cache_stats["stored"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total * 100) if total > 0 else 0
        
        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "stored": self.cache_stats["stored"],
            "hit_rate": f"{hit_rate:.1f}%"
        }

# 全局缓存实例
_smart_cache = SmartCache()

def cache_result(prefix: str, ttl_hours: int = 1):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 尝试从缓存获取
            cache_data = {
                "args": args,
                "kwargs": kwargs
            }
            
            cached_result = _smart_cache.get(prefix, cache_data, ttl_hours)
            if cached_result is not None:
                print(f"[缓存命中] {prefix}")
                return cached_result
            
            # 执行函数
            print(f"[执行函数] {prefix}")
            result = func(*args, **kwargs)
            
            # 存储到缓存
            _smart_cache.set(prefix, cache_data, result, ttl_hours)
            print(f"[结果已缓存] {prefix}")
            
            return result
        return wrapper
    return decorator

def auto_cache_command(command_func):
    """自动缓存命令装饰器"""
    @wraps(command_func)
    def wrapper(*args, **kwargs):
        if not AUTO_CACHE_AVAILABLE:
            return command_func(*args, **kwargs)
        
        # 提取命令信息
        command_name = command_func.__name__
        command_args = args[1:] if len(args) > 1 else []
        
        # 检查缓存
        cache_key = f"cmd_{command_name}"
        cached_result = _smart_cache.get(cache_key, command_args)
        
        if cached_result:
            print(f"[命令缓存命中] {command_name}")
            return cached_result
        
        # 执行命令
        print(f"[执行命令] {command_name}")
        start_time = time.time()
        result = command_func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # 添加执行信息
        result_with_meta = {
            "result": result,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "command": command_name,
            "args": command_args
        }
        
        # 缓存结果（根据执行时间决定TTL）
        ttl_hours = 24 if execution_time > 5 else 1  # 慢命令缓存更久
        _smart_cache.set(cache_key, command_args, result_with_meta, ttl_hours)
        
        print(f"[命令完成] ({execution_time:.2f}s)，已缓存")
        
        return result
    
    return wrapper

def cache_report(report_data: Dict[str, Any], report_type: str = "general"):
    """缓存报告"""
    if AUTO_CACHE_AVAILABLE:
        # 生成报告ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = f"{report_type}_{timestamp}"
        
        # 添加元数据
        report_data["_cache_meta"] = {
            "report_id": report_id,
            "report_type": report_type,
            "created_at": datetime.now().isoformat(),
            "auto_cached": True
        }
        
        # 使用缓存管理器存储
        cache_manager = get_auto_manager()
        file_path = cache_manager.store_report(report_data, report_type)
        
        print(f"[报告已缓存] {report_id}")
        return report_id
    
    return None

def get_cache_info():
    """获取缓存信息"""
    if not AUTO_CACHE_AVAILABLE:
        return {"error": "自动缓存不可用"}
    
    cache_manager = get_auto_manager()
    status = cache_manager.get_status()
    smart_stats = _smart_cache.get_stats()
    
    return {
        "auto_manager": status,
        "smart_cache": smart_stats,
        "available": True
    }

def cleanup_cache(force: bool = False):
    """清理缓存"""
    if AUTO_CACHE_AVAILABLE:
        cache_manager = get_auto_manager()
        result = cache_manager._perform_auto_cleanup() if force else cache_manager._perform_auto_cleanup()
        return result
    return {"error": "自动缓存不可用"}

# 示例：为ARQ分析添加缓存支持
@cache_result("arq_analysis", ttl_hours=24)
def cached_arq_analysis(query: str) -> Dict[str, Any]:
    """缓存的ARQ分析"""
    # 这里会调用实际的ARQ分析逻辑
    # 模拟分析过程
    time.sleep(0.5)  # 模拟处理时间
    
    return {
        "query": query,
        "analysis": f"这是对'{query}'的分析结果",
        "timestamp": datetime.now().isoformat(),
        "confidence": 0.85
    }

if __name__ == "__main__":
    # 测试缓存功能
    print("测试智能缓存系统...")
    
    # 测试ARQ分析缓存
    query = "什么是ARQ推理引擎？"
    print(f"\n查询: {query}")
    
    # 第一次执行（会缓存）
    result1 = cached_arq_analysis(query)
    print(f"结果: {result1['analysis']}")
    
    # 第二次执行（从缓存获取）
    result2 = cached_arq_analysis(query)
    print(f"结果: {result2['analysis']}")
    
    # 显示缓存统计
    print("\n缓存统计:")
    stats = get_cache_info()
    if 'smart_cache' in stats:
        print(f"  命中率: {stats['smart_cache']['hit_rate']}")
        print(f"  命中次数: {stats['smart_cache']['hits']}")
        print(f"  未命中次数: {stats['smart_cache']['misses']}")
    else:
        print("  缓存统计不可用")
    
    # 清理测试
    print("\n清理缓存...")
    cleanup_result = cleanup_cache()
    print(f"清理完成: {cleanup_result}")