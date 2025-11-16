#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用记录追踪系统
记录知识库的详细使用情况，包括用户、操作、文件访问等
"""

import os
import json
import time
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import uuid

# 项目根路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
USAGE_TRACKER_ROOT = PROJECT_ROOT / "knowledge_base" / "usage_tracker"

@dataclass
class UsageConfig:
    """使用记录配置"""
    user_id: str
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    execution_time: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class FileAccessConfig:
    """文件访问配置"""
    user_id: str
    session_id: str
    file_path: str
    file_id: str
    group_id: str
    action: str
    query: Optional[str] = None
    relevance_score: Optional[float] = None
    access_count: int = 1

@dataclass
class UsageRecord:
    """使用记录"""
    record_id: str
    timestamp: datetime
    user_id: str
    session_id: str
    action: str
    resource_type: str  # document, group, search, ai_enhance, etc.
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    execution_time: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class FileAccessRecord:
    """文件访问记录"""
    record_id: str
    timestamp: datetime
    user_id: str
    session_id: str
    file_path: str
    file_id: str
    group_id: str
    action: str  # read, search, download, etc.
    query: Optional[str] = None
    relevance_score: Optional[float] = None
    access_count: int = 1

class UsageTracker:
    """使用记录追踪器"""
    
    def __init__(self):
        self.usage_tracker_root = USAGE_TRACKER_ROOT
        self.usage_tracker_root.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.usage_tracker_root / "records").mkdir(exist_ok=True)
        (self.usage_tracker_root / "sessions").mkdir(exist_ok=True)
        (self.usage_tracker_root / "analytics").mkdir(exist_ok=True)
        
        # 内存缓存
        self.records_cache = deque(maxlen=10000)  # 最近10000条记录
        self.file_access_cache = deque(maxlen=5000)  # 最近5000条文件访问记录
        self.session_cache = {}  # 活跃会话
        
        # 统计数据
        self.stats = {
            "total_records": 0,
            "total_sessions": 0,
            "active_sessions": 0,
            "most_accessed_files": {},
            "most_active_users": {},
            "action_counts": defaultdict(int),
            "hourly_activity": defaultdict(int)
        }
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 启动后台任务
        self._start_background_tasks()
    
    def track_usage(self, config: UsageConfig) -> str:
        """记录使用情况
        
        Args:
            config: 使用记录配置对象
            
        Returns:
            str: 记录ID
        """
        # 获取或创建会话
        session_id = self._get_or_create_session(config.user_id)
        
        # 创建记录
        record = UsageRecord(
            record_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=config.user_id,
            session_id=session_id,
            action=config.action,
            resource_type=config.resource_type,
            resource_id=config.resource_id,
            resource_name=config.resource_name,
            details=config.details or {},
            ip_address=config.ip_address,
            user_agent=config.user_agent,
            execution_time=config.execution_time,
            success=config.success,
            error_message=config.error_message
        )
        
        # 添加到缓存
        with self.lock:
            self.records_cache.append(record)
            self.stats["total_records"] += 1
            self.stats["action_counts"][action] += 1
            
            # 更新小时活动统计
            hour_key = record.timestamp.strftime("%Y-%m-%d %H:00")
            self.stats["hourly_activity"][hour_key] += 1
            
            # 更新用户活动统计
            if config.user_id not in self.stats["most_active_users"]:
                self.stats["most_active_users"][config.user_id] = 0
            self.stats["most_active_users"][config.user_id] += 1
        
        # 异步保存到文件
        self._save_record_async(record)
        
        return record.record_id
    
    def track_file_access(self, config: FileAccessConfig) -> str:
        """记录文件访问
        
        Args:
            config: 文件访问配置对象
            
        Returns:
            str: 记录ID
        """
        
        session_id = self._get_or_create_session(config.user_id)
        
        # 检查是否已有相同访问记录
        existing_record = None
        with self.lock:
            for record in reversed(self.file_access_cache):
                if (record.user_id == config.user_id and 
                    record.file_id == config.file_id and 
                    record.action == config.action and
                    record.timestamp.date() == datetime.now().date()):
                    existing_record = record
                    break
        
        if existing_record:
            # 更新现有记录
            with self.lock:
                existing_record.access_count += 1
                existing_record.timestamp = datetime.now()
                if config.query:
                    existing_record.query = config.query
                if config.relevance_score:
                    existing_record.relevance_score = config.relevance_score
            record_id = existing_record.record_id
        else:
            # 创建新记录
            record = FileAccessRecord(
                record_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                user_id=user_id,
                session_id=session_id,
                file_path=file_path,
                file_id=file_id,
                group_id=group_id,
                action=action,
                query=query,
                relevance_score=relevance_score
            )
            
            with self.lock:
                self.file_access_cache.append(record)
                
                # 更新最常访问文件统计
                file_key = f"{group_id}:{file_id}"
                if file_key not in self.stats["most_accessed_files"]:
                    self.stats["most_accessed_files"][file_key] = 0
                self.stats["most_accessed_files"][file_key] += 1
            
            # 异步保存
            self._save_file_access_async(record)
            record_id = record.record_id
        
        return record_id
    
    def get_usage_records(self,
                         user_id: Optional[str] = None,
                         action: Optional[str] = None,
                         resource_type: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """获取使用记录"""
        
        records = []
        with self.lock:
            for record in reversed(self.records_cache):
                # 应用过滤条件
                if user_id and record.user_id != user_id:
                    continue
                if action and record.action != action:
                    continue
                if resource_type and record.resource_type != resource_type:
                    continue
                if start_time and record.timestamp < start_time:
                    continue
                if end_time and record.timestamp > end_time:
                    continue
                
                # 转换为字典
                record_dict = asdict(record)
                record_dict['timestamp'] = record.timestamp.isoformat()
                records.append(record_dict)
                
                if len(records) >= limit:
                    break
        
        return records
    
    def get_file_access_records(self,
                               user_id: Optional[str] = None,
                               group_id: Optional[str] = None,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """获取文件访问记录"""
        
        records = []
        with self.lock:
            for record in reversed(self.file_access_cache):
                # 应用过滤条件
                if user_id and record.user_id != user_id:
                    continue
                if group_id and record.group_id != group_id:
                    continue
                if start_time and record.timestamp < start_time:
                    continue
                if end_time and record.timestamp > end_time:
                    continue
                
                # 转换为字典
                record_dict = asdict(record)
                record_dict['timestamp'] = record.timestamp.isoformat()
                records.append(record_dict)
                
                if len(records) >= limit:
                    break
        
        return records
    
    def get_analytics(self) -> Dict[str, Any]:
        """获取分析数据"""
        with self.lock:
            # 计算活跃会话数
            current_time = datetime.now()
            active_sessions = 0
            for session_id, session_data in self.session_cache.items():
                last_activity = session_data.get('last_activity')
                if last_activity and (current_time - last_activity).seconds < 1800:  # 30分钟内活跃
                    active_sessions += 1
            
            self.stats["active_sessions"] = active_sessions
            
            # 获取最常访问的文件（前10）
            most_accessed = sorted(
                self.stats["most_accessed_files"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # 获取最活跃的用户（前10）
            most_active = sorted(
                self.stats["most_active_users"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # 获取最近24小时的活动
            recent_activity = defaultdict(int)
            now = datetime.now()
            for i in range(24):
                hour_key = (now - timedelta(hours=i)).strftime("%Y-%m-%d %H:00")
                recent_activity[hour_key] = self.stats["hourly_activity"].get(hour_key, 0)
            
            return {
                "summary": {
                    "total_records": self.stats["total_records"],
                    "total_sessions": self.stats["total_sessions"],
                    "active_sessions": active_sessions,
                    "total_file_accesses": len(self.file_access_cache)
                },
                "most_accessed_files": most_accessed,
                "most_active_users": most_active,
                "action_distribution": dict(self.stats["action_counts"]),
                "hourly_activity": dict(recent_activity),
                "cache_sizes": {
                    "usage_records": len(self.records_cache),
                    "file_access_records": len(self.file_access_cache),
                    "sessions": len(self.session_cache)
                }
            }
    
    def get_user_activity(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """获取用户活动详情"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # 获取用户记录
        records = self.get_usage_records(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        # 获取文件访问记录
        file_records = self.get_file_access_records(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        # 统计分析
        action_counts = defaultdict(int)
        resource_type_counts = defaultdict(int)
        daily_activity = defaultdict(int)
        accessed_files = defaultdict(int)
        
        for record in records:
            action_counts[record["action"]] += 1
            resource_type_counts[record["resource_type"]] += 1
            day_key = record["timestamp"][:10]
            daily_activity[day_key] += 1
        
        for record in file_records:
            file_key = f"{record['group_id']}:{record['file_id']}"
            accessed_files[file_key] += record["access_count"]
        
        return {
            "user_id": user_id,
            "period": f"{days} days",
            "total_actions": len(records),
            "total_file_accesses": len(file_records),
            "action_counts": dict(action_counts),
            "resource_type_counts": dict(resource_type_counts),
            "daily_activity": dict(daily_activity),
            "most_accessed_files": sorted(accessed_files.items(), key=lambda x: x[1], reverse=True)[:10],
            "recent_records": records[:10],
            "recent_file_accesses": file_records[:10]
        }
    
    def _get_or_create_session(self, user_id: str) -> str:
        """获取或创建会话"""
        current_time = datetime.now()
        
        with self.lock:
            # 查找现有会话
            for session_id, session_data in self.session_cache.items():
                if (session_data["user_id"] == user_id and 
                    (current_time - session_data["last_activity"]).seconds < 1800):  # 30分钟内
                    session_data["last_activity"] = current_time
                    session_data["activity_count"] += 1
                    return session_id
            
            # 创建新会话
            session_id = str(uuid.uuid4())
            self.session_cache[session_id] = {
                "user_id": user_id,
                "created_at": current_time,
                "last_activity": current_time,
                "activity_count": 1
            }
            self.stats["total_sessions"] += 1
            
            # 保存会话信息
            self._save_session_async(session_id, self.session_cache[session_id])
            
            return session_id
    
    def _save_record_async(self, record: UsageRecord):
        """异步保存记录"""
        def save():
            try:
                # 按日期保存
                date_str = record.timestamp.strftime("%Y-%m-%d")
                record_file = self.usage_tracker_root / "records" / f"usage_{date_str}.jsonl"
                
                record_data = asdict(record)
                record_data['timestamp'] = record.timestamp.isoformat()
                
                with open(record_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record_data, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"保存使用记录失败: {e}")
        
        threading.Thread(target=save, daemon=True).start()
    
    def _save_file_access_async(self, record: FileAccessRecord):
        """异步保存文件访问记录"""
        def save():
            try:
                # 按日期保存
                date_str = record.timestamp.strftime("%Y-%m-%d")
                record_file = self.usage_tracker_root / "records" / f"file_access_{date_str}.jsonl"
                
                record_data = asdict(record)
                record_data['timestamp'] = record.timestamp.isoformat()
                
                with open(record_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record_data, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"保存文件访问记录失败: {e}")
        
        threading.Thread(target=save, daemon=True).start()
    
    def _save_session_async(self, session_id: str, session_data: Dict[str, Any]):
        """异步保存会话信息"""
        def save():
            try:
                session_file = self.usage_tracker_root / "sessions" / f"{session_id}.json"
                
                save_data = session_data.copy()
                save_data["created_at"] = save_data["created_at"].isoformat()
                save_data["last_activity"] = save_data["last_activity"].isoformat()
                
                with open(session_file, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"保存会话信息失败: {e}")
        
        threading.Thread(target=save, daemon=True).start()
    
    def _start_background_tasks(self):
        """启动后台任务"""
        def cleanup_task():
            """清理过期数据"""
            while True:
                try:
                    time.sleep(3600)  # 每小时执行一次
                    
                    current_time = datetime.now()
                    
                    # 清理过期会话
                    with self.lock:
                        expired_sessions = []
                        for session_id, session_data in self.session_cache.items():
                            if (current_time - session_data["last_activity"]).seconds > 86400:  # 24小时
                                expired_sessions.append(session_id)
                        
                        for session_id in expired_sessions:
                            del self.session_cache[session_id]
                            # 删除会话文件
                            session_file = self.usage_tracker_root / "sessions" / f"{session_id}.json"
                            if session_file.exists():
                                session_file.unlink()
                    
                    # 保存分析数据
                    self._save_analytics()
                    
                except Exception as e:
                    print(f"后台清理任务失败: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
    
    def _save_analytics(self):
        """保存分析数据"""
        try:
            analytics_file = self.usage_tracker_root / "analytics" / f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            analytics_data = self.get_analytics()
            
            with open(analytics_file, 'w', encoding='utf-8') as f:
                json.dump(analytics_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存分析数据失败: {e}")

# 全局实例
_usage_tracker = None

def get_usage_tracker() -> UsageTracker:
    """获取使用记录追踪器实例"""
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = UsageTracker()
    return _usage_tracker

# 便捷函数
def track_usage(user_id: str, action: str, resource_type: str, **kwargs) -> str:
    """记录使用情况的便捷函数"""
    tracker = get_usage_tracker()
    return tracker.track_usage(user_id, action, resource_type, **kwargs)

def track_file_access(user_id: str, file_path: str, file_id: str, group_id: str, **kwargs) -> str:
    """记录文件访问的便捷函数"""
    tracker = get_usage_tracker()
    return tracker.track_file_access(user_id, file_path, file_id, group_id, **kwargs)
