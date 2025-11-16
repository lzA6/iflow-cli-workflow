#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 ARQ数据管理器 V17 Hyperdimensional Singularity
=================================================

这是ARQ系统的综合数据管理器，实现自动化的数据读取、调用、记录、总结和查看功能：
- 🔄 自动数据读取和调用机制
- 📊 智能数据记录和分析
- 🧠 会话历史管理
- 💾 知识库自动同步
- 🎯 偏好数据学习
- 📈 数据趋势分析
- 🔍 智能检索系统
- 🛡️ 数据安全保障

核心特性：
- 全自动数据处理流程
- 智能缓存机制
- 实时数据同步
- 深度学习用户偏好
- 高效检索算法
- 数据持久化
- 跨会话数据连续性

作者: AI架构师团队
版本: 17.0.0 Hyperdimensional Singularity
日期: 2025-11-17
"""

import os
import sys
import json
import sqlite3
import asyncio
import logging
import time
import uuid
import hashlib
import threading
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import warnings

# 项目根路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入现有组件
try:
    from session_cache_manager import get_缓存管理器
    from memory_optimizer import get_memory_optimizer
    from arq_reasoning_engine_v17_hyperdimensional_singularity import ARQReasoningEngineV17HyperdimensionalSingularity
    LEGACY_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 部分传统组件不可用: {e}")
    LEGACY_COMPONENTS_AVAILABLE = False

# 抑制警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据类型枚举
class DataType(Enum):
    """数据类型枚举"""
    SESSION_DATA = "session_data"
    KNOWLEDGE_BASE = "knowledge_base"
    USER_PREFERENCES = "user_preferences"
    QUERY_HISTORY = "query_history"
    ARQ_HISTORY = "arq_history"
    SYSTEM_METRICS = "system_metrics"
    CACHE_DATA = "cache_data"
    MEMORY_SNAPSHOT = "memory_snapshot"

# 数据优先级
class DataPriority(Enum):
    """数据优先级"""
    CRITICAL = 1    # 关键数据，永不删除
    HIGH = 2        # 高优先级，优先保留
    NORMAL = 3      # 普通优先级
    LOW = 4         # 低优先级，优先清理

# 数据项结构
@dataclass
class DataItem:
    """数据项结构"""
    id: str
    data_type: DataType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    priority: DataPriority = DataPriority.NORMAL
    tags: Set[str] = field(default_factory=set)
    size_bytes: int = 0
    checksum: str = ""

# 会话上下文
@dataclass
class SessionContext:
    """会话上下文"""
    session_id: str
    project_id: str
    user_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    goals: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    context_data: Dict[str, Any] = field(default_factory=dict)
    active: bool = True

# 用户偏好
@dataclass
class UserPreferences:
    """用户偏好设置"""
    user_id: str
    preferred_thinking_mode: str = "hyperdimensional_singularity"
    language_preference: str = "zh-CN"
    response_style: str = "professional"
    auto_save_frequency: int = 300  # 秒
    cache_retention_days: int = 30
    privacy_level: str = "standard"
    notification_settings: Dict[str, bool] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

class ARQDataManagerV17:
    """ARQ数据管理器V17主类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化数据管理器"""
        self.config = config or {}
        
        # 数据目录
        self.data_root = PROJECT_ROOT / "data"
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # 数据库路径
        self.main_db_path = self.data_root / "arq_data_manager.db"
        self.cache_db_path = self.data_root / "cache" / "cache.db"
        
        # 初始化数据库
        self._init_databases()
        
        # 内存缓存
        self.memory_cache = {}
        self.cache_lock = threading.RLock()
        
        # 会话管理
        self.active_sessions = {}
        self.session_lock = threading.RLock()
        
        # 用户偏好
        self.user_preferences = {}
        
        # 性能监控
        self.performance_metrics = {
            "data_reads": 0,
            "data_writes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "query_count": 0,
            "session_count": 0
        }
        
        # 自动同步线程
        self.sync_thread = None
        self.sync_interval = 60  # 秒
        self.running = False
        
        # 传统组件集成
        self.legacy_cache_manager = None
        self.memory_optimizer = None
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 初始化传统组件
        self._init_legacy_components()
        
        logger.info("🌟 ARQ数据管理器V17初始化完成")
    
    def _init_databases(self):
        """初始化数据库"""
        # 主数据库
        with sqlite3.connect(self.main_db_path) as conn:
            # 数据项表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_items (
                    id TEXT PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    priority INTEGER DEFAULT 3,
                    tags TEXT,
                    size_bytes INTEGER DEFAULT 0,
                    checksum TEXT
                )
            """)
            
            # 会话表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    user_id TEXT,
                    start_time TEXT NOT NULL,
                    goals TEXT,
                    achievements TEXT,
                    blockers TEXT,
                    preferences TEXT,
                    context_data TEXT,
                    active INTEGER DEFAULT 1
                )
            """)
            
            # 用户偏好表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferred_thinking_mode TEXT,
                    language_preference TEXT,
                    response_style TEXT,
                    auto_save_frequency INTEGER,
                    cache_retention_days INTEGER,
                    privacy_level TEXT,
                    notification_settings TEXT,
                    custom_settings TEXT,
                    last_updated TEXT
                )
            """)
            
            # 查询历史表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    query TEXT NOT NULL,
                    context TEXT,
                    response TEXT,
                    timestamp TEXT NOT NULL,
                    response_time REAL,
                    confidence REAL,
                    metadata TEXT
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_data_type ON data_items(data_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON data_items(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON data_items(last_accessed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON sessions(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_timestamp ON query_history(timestamp)")
            
            conn.commit()
        
        logger.info("✅ 数据库初始化完成")
    
    def _init_legacy_components(self):
        """初始化传统组件"""
        try:
            if LEGACY_COMPONENTS_AVAILABLE:
                self.legacy_cache_manager = get_缓存管理器()
                self.memory_optimizer = get_memory_optimizer()
                logger.info("✅ 传统组件集成成功")
        except Exception as e:
            logger.warning(f"⚠️ 传统组件集成失败: {e}")
    
    async def start_auto_sync(self):
        """启动自动同步"""
        if self.running:
            return
        
        self.running = True
        self.sync_thread = threading.Thread(target=self._auto_sync_worker, daemon=True)
        self.sync_thread.start()
        logger.info("🔄 自动同步已启动")
    
    def stop_auto_sync(self):
        """停止自动同步"""
        self.running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5)
        logger.info("⏹️ 自动同步已停止")
    
    def _auto_sync_worker(self):
        """自动同步工作线程"""
        while self.running:
            try:
                asyncio.run(self._perform_sync())
                time.sleep(self.sync_interval)
            except Exception as e:
                logger.error(f"❌ 自动同步失败: {e}")
                time.sleep(10)
    
    async def _perform_sync(self):
        """执行同步操作"""
        # 同步内存缓存到数据库
        await self._sync_memory_to_db()
        
        # 清理过期数据
        await self._cleanup_expired_data()
        
        # 优化数据库
        await self._optimize_database()
        
        # 更新性能指标
        await self._update_performance_metrics()
    
    async def store_data(self, data: Any, data_type: DataType, 
                        metadata: Optional[Dict] = None,
                        priority: DataPriority = DataPriority.NORMAL,
                        tags: Optional[Set[str]] = None) -> str:
        """
        存储数据
        
        Args:
            data: 要存储的数据
            data_type: 数据类型
            metadata: 元数据
            priority: 优先级
            tags: 标签集合
            
        Returns:
            数据ID
        """
        try:
            # 生成数据ID
            data_id = str(uuid.uuid4())
            
            # 序列化数据
            content = json.dumps(data, ensure_ascii=False, default=str)
            content_bytes = content.encode('utf-8')
            
            # 计算校验和
            checksum = hashlib.md5(content_bytes).hexdigest()
            
            # 创建数据项
            data_item = DataItem(
                id=data_id,
                data_type=data_type,
                content=data,
                metadata=metadata or {},
                priority=priority,
                tags=tags or set(),
                size_bytes=len(content_bytes),
                checksum=checksum
            )
            
            # 存储到内存缓存
            with self.cache_lock:
                self.memory_cache[data_id] = data_item
            
            # 存储到数据库
            await self._store_to_db(data_item)
            
            # 存储到传统缓存
            if self.legacy_cache_manager:
                await self.legacy_cache_manager.设置缓存(
                    f"arq_data_{data_id}", 
                    data_item,
                    timedelta(days=30)
                )
            
            # 更新性能指标
            self.performance_metrics["data_writes"] += 1
            
            logger.debug(f"💾 数据已存储: {data_id} ({data_type.value})")
            return data_id
            
        except Exception as e:
            logger.error(f"❌ 存储数据失败: {e}")
            raise
    
    async def retrieve_data(self, data_id: str) -> Optional[DataItem]:
        """
        检索数据
        
        Args:
            data_id: 数据ID
            
        Returns:
            数据项或None
        """
        try:
            # 首先检查内存缓存
            with self.cache_lock:
                if data_id in self.memory_cache:
                    data_item = self.memory_cache[data_id]
                    data_item.last_accessed = datetime.now()
                    data_item.access_count += 1
                    self.performance_metrics["cache_hits"] += 1
                    return data_item
            
            # 检查传统缓存
            if self.legacy_cache_manager:
                cached_data = await self.legacy_cache_manager.获取缓存(f"arq_data_{data_id}")
                if cached_data:
                    # 转换为DataItem
                    data_item = DataItem(**cached_data)
                    data_item.last_accessed = datetime.now()
                    data_item.access_count += 1
                    
                    # 加载到内存缓存
                    with self.cache_lock:
                        self.memory_cache[data_id] = data_item
                    
                    self.performance_metrics["cache_hits"] += 1
                    return data_item
            
            # 从数据库加载
            data_item = await self._load_from_db(data_id)
            if data_item:
                data_item.last_accessed = datetime.now()
                data_item.access_count += 1
                
                # 加载到内存缓存
                with self.cache_lock:
                    self.memory_cache[data_id] = data_item
                
                self.performance_metrics["cache_misses"] += 1
                return data_item
            
            self.performance_metrics["cache_misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"❌ 检索数据失败 {data_id}: {e}")
            return None
    
    async def create_session(self, project_id: str, user_id: Optional[str] = None,
                           goals: Optional[List[str]] = None) -> str:
        """
        创建新会话
        
        Args:
            project_id: 项目ID
            user_id: 用户ID
            goals: 会话目标
            
        Returns:
            会话ID
        """
        try:
            session_id = str(uuid.uuid4())
            
            session = SessionContext(
                session_id=session_id,
                project_id=project_id,
                user_id=user_id,
                goals=goals or []
            )
            
            # 存储会话
            with self.session_lock:
                self.active_sessions[session_id] = session
            
            # 存储到数据库
            await self._store_session_to_db(session)
            
            # 更新性能指标
            self.performance_metrics["session_count"] += 1
            
            logger.info(f"🆕 会话已创建: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"❌ 创建会话失败: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[SessionContext]:
        """获取会话上下文"""
        try:
            # 检查活跃会话
            with self.session_lock:
                if session_id in self.active_sessions:
                    return self.active_sessions[session_id]
            
            # 从数据库加载
            session = await self._load_session_from_db(session_id)
            if session:
                with self.session_lock:
                    self.active_sessions[session_id] = session
            
            return session
            
        except Exception as e:
            logger.error(f"❌ 获取会话失败 {session_id}: {e}")
            return None
    
    async def update_session(self, session_id: str, **kwargs) -> bool:
        """更新会话上下文"""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            # 更新属性
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            
            # 存储到数据库
            await self._store_session_to_db(session)
            
            logger.debug(f"📝 会话已更新: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 更新会话失败 {session_id}: {e}")
            return False
    
    async def record_query(self, session_id: str, query: str, 
                          context: Optional[str] = None,
                          response: Optional[str] = None,
                          response_time: Optional[float] = None,
                          confidence: Optional[float] = None,
                          metadata: Optional[Dict] = None) -> str:
        """
        记录查询
        
        Args:
            session_id: 会话ID
            query: 查询内容
            context: 上下文
            response: 响应内容
            response_time: 响应时间
            confidence: 置信度
            metadata: 元数据
            
        Returns:
            查询ID
        """
        try:
            query_id = str(uuid.uuid4())
            
            # 存储到数据库
            with sqlite3.connect(self.main_db_path) as conn:
                conn.execute("""
                    INSERT INTO query_history 
                    (id, session_id, query, context, response, timestamp, response_time, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_id,
                    session_id,
                    query,
                    context,
                    response,
                    datetime.now().isoformat(),
                    response_time,
                    confidence,
                    json.dumps(metadata or {}, ensure_ascii=False)
                ))
                conn.commit()
            
            # 更新性能指标
            self.performance_metrics["query_count"] += 1
            
            logger.debug(f"📝 查询已记录: {query_id}")
            return query_id
            
        except Exception as e:
            logger.error(f"❌ 记录查询失败: {e}")
            raise
    
    async def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """获取用户偏好"""
        try:
            # 检查内存缓存
            if user_id in self.user_preferences:
                return self.user_preferences[user_id]
            
            # 从数据库加载
            with sqlite3.connect(self.main_db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM user_preferences WHERE user_id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    preferences = UserPreferences(
                        user_id=row[0],
                        preferred_thinking_mode=row[1] or "hyperdimensional_singularity",
                        language_preference=row[2] or "zh-CN",
                        response_style=row[3] or "professional",
                        auto_save_frequency=row[4] or 300,
                        cache_retention_days=row[5] or 30,
                        privacy_level=row[6] or "standard",
                        notification_settings=json.loads(row[7] or "{}"),
                        custom_settings=json.loads(row[8] or "{}"),
                        last_updated=datetime.fromisoformat(row[9]) if row[9] else datetime.now()
                    )
                    
                    self.user_preferences[user_id] = preferences
                    return preferences
            
            # 创建默认偏好
            preferences = UserPreferences(user_id=user_id)
            await self.save_user_preferences(preferences)
            
            return preferences
            
        except Exception as e:
            logger.error(f"❌ 获取用户偏好失败 {user_id}: {e}")
            return None
    
    async def save_user_preferences(self, preferences: UserPreferences) -> bool:
        """保存用户偏好"""
        try:
            preferences.last_updated = datetime.now()
            
            with sqlite3.connect(self.main_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO user_preferences 
                    (user_id, preferred_thinking_mode, language_preference, response_style,
                     auto_save_frequency, cache_retention_days, privacy_level,
                     notification_settings, custom_settings, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    preferences.user_id,
                    preferences.preferred_thinking_mode,
                    preferences.language_preference,
                    preferences.response_style,
                    preferences.auto_save_frequency,
                    preferences.cache_retention_days,
                    preferences.privacy_level,
                    json.dumps(preferences.notification_settings, ensure_ascii=False),
                    json.dumps(preferences.custom_settings, ensure_ascii=False),
                    preferences.last_updated.isoformat()
                ))
                conn.commit()
            
            # 更新内存缓存
            self.user_preferences[preferences.user_id] = preferences
            
            logger.debug(f"💾 用户偏好已保存: {preferences.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 保存用户偏好失败: {e}")
            return False
    
    async def search_data(self, query: str, data_types: Optional[List[DataType]] = None,
                         tags: Optional[Set[str]] = None,
                         limit: int = 100) -> List[DataItem]:
        """
        搜索数据
        
        Args:
            query: 搜索查询
            data_types: 数据类型过滤
            tags: 标签过滤
            limit: 结果限制
            
        Returns:
            匹配的数据项列表
        """
        try:
            results = []
            
            # 构建SQL查询
            sql_conditions = []
            sql_params = []
            
            if data_types:
                type_placeholders = ",".join(["?" for _ in data_types])
                sql_conditions.append(f"data_type IN ({type_placeholders})")
                sql_params.extend([dt.value for dt in data_types])
            
            if tags:
                for tag in tags:
                    sql_conditions.append("tags LIKE ?")
                    sql_params.append(f"%{tag}%")
            
            where_clause = " AND ".join(sql_conditions) if sql_conditions else "1=1"
            
            with sqlite3.connect(self.main_db_path) as conn:
                cursor = conn.execute(f"""
                    SELECT * FROM data_items 
                    WHERE {where_clause}
                    ORDER BY last_accessed DESC
                    LIMIT ?
                """, sql_params + [limit])
                
                rows = cursor.fetchall()
                
                for row in rows:
                    try:
                        data_item = DataItem(
                            id=row[0],
                            data_type=DataType(row[1]),
                            content=json.loads(row[2]),
                            metadata=json.loads(row[3] or "{}"),
                            created_at=datetime.fromisoformat(row[4]),
                            last_accessed=datetime.fromisoformat(row[5]),
                            access_count=row[6],
                            priority=DataPriority(row[7]),
                            tags=set(json.loads(row[8] or "[]")),
                            size_bytes=row[9],
                            checksum=row[10]
                        )
                        
                        # 简单的内容匹配
                        if query.lower() in json.dumps(data_item.content, ensure_ascii=False).lower():
                            results.append(data_item)
                            
                    except Exception as e:
                        logger.warning(f"⚠️ 解析数据项失败: {e}")
                        continue
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 搜索数据失败: {e}")
            return []
    
    async def get_session_history(self, session_id: str, limit: int = 100) -> List[Dict]:
        """获取会话历史"""
        try:
            with sqlite3.connect(self.main_db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM query_history 
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (session_id, limit))
                
                rows = cursor.fetchall()
                
                history = []
                for row in rows:
                    history.append({
                        "id": row[0],
                        "session_id": row[1],
                        "query": row[2],
                        "context": row[3],
                        "response": row[4],
                        "timestamp": row[5],
                        "response_time": row[6],
                        "confidence": row[7],
                        "metadata": json.loads(row[8] or "{}")
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"❌ 获取会话历史失败 {session_id}: {e}")
            return []
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        try:
            summary = {
                "performance_metrics": self.performance_metrics.copy(),
                "cache_hit_rate": 0.0,
                "active_sessions": len(self.active_sessions),
                "memory_usage": 0,
                "database_size": 0
            }
            
            # 计算缓存命中率
            total_requests = summary["performance_metrics"]["cache_hits"] + summary["performance_metrics"]["cache_misses"]
            if total_requests > 0:
                summary["cache_hit_rate"] = summary["performance_metrics"]["cache_hits"] / total_requests
            
            # 获取内存使用情况
            if self.memory_optimizer:
                stats = self.memory_optimizer.get_memory_stats()
                summary["memory_usage"] = stats.process_mb
            
            # 获取数据库大小
            try:
                summary["database_size"] = os.path.getsize(self.main_db_path) / (1024 * 1024)  # MB
            except:
                pass
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ 获取性能摘要失败: {e}")
            return {}
    
    async def _store_to_db(self, data_item: DataItem):
        """存储数据项到数据库"""
        with sqlite3.connect(self.main_db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO data_items 
                (id, data_type, content, metadata, created_at, last_accessed, 
                 access_count, priority, tags, size_bytes, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data_item.id,
                data_item.data_type.value,
                json.dumps(data_item.content, ensure_ascii=False, default=str),
                json.dumps(data_item.metadata, ensure_ascii=False),
                data_item.created_at.isoformat(),
                data_item.last_accessed.isoformat(),
                data_item.access_count,
                data_item.priority.value,
                json.dumps(list(data_item.tags), ensure_ascii=False),
                data_item.size_bytes,
                data_item.checksum
            ))
            conn.commit()
    
    async def _load_from_db(self, data_id: str) -> Optional[DataItem]:
        """从数据库加载数据项"""
        with sqlite3.connect(self.main_db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM data_items WHERE id = ?
            """, (data_id,))
            
            row = cursor.fetchone()
            if row:
                return DataItem(
                    id=row[0],
                    data_type=DataType(row[1]),
                    content=json.loads(row[2]),
                    metadata=json.loads(row[3] or "{}"),
                    created_at=datetime.fromisoformat(row[4]),
                    last_accessed=datetime.fromisoformat(row[5]),
                    access_count=row[6],
                    priority=DataPriority(row[7]),
                    tags=set(json.loads(row[8] or "[]")),
                    size_bytes=row[9],
                    checksum=row[10]
                )
        
        return None
    
    async def _store_session_to_db(self, session: SessionContext):
        """存储会话到数据库"""
        with sqlite3.connect(self.main_db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions 
                (session_id, project_id, user_id, start_time, goals, achievements,
                 blockers, preferences, context_data, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.project_id,
                session.user_id,
                session.start_time.isoformat(),
                json.dumps(session.goals, ensure_ascii=False),
                json.dumps(session.achievements, ensure_ascii=False),
                json.dumps(session.blockers, ensure_ascii=False),
                json.dumps(session.preferences, ensure_ascii=False),
                json.dumps(session.context_data, ensure_ascii=False),
                int(session.active)
            ))
            conn.commit()
    
    async def _load_session_from_db(self, session_id: str) -> Optional[SessionContext]:
        """从数据库加载会话"""
        with sqlite3.connect(self.main_db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM sessions WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if row:
                return SessionContext(
                    session_id=row[0],
                    project_id=row[1],
                    user_id=row[2],
                    start_time=datetime.fromisoformat(row[3]),
                    goals=json.loads(row[4] or "[]"),
                    achievements=json.loads(row[5] or "[]"),
                    blockers=json.loads(row[6] or "[]"),
                    preferences=json.loads(row[7] or "{}"),
                    context_data=json.loads(row[8] or "{}"),
                    active=bool(row[9])
                )
        
        return None
    
    async def _sync_memory_to_db(self):
        """同步内存缓存到数据库"""
        try:
            with self.cache_lock:
                items_to_sync = list(self.memory_cache.values())
            
            for item in items_to_sync:
                await self._store_to_db(item)
                
            logger.debug("💾 内存缓存已同步到数据库")
            
        except Exception as e:
            logger.error(f"❌ 同步内存到数据库失败: {e}")
    
    async def _cleanup_expired_data(self):
        """清理过期数据"""
        try:
            cutoff_date = datetime.now() - timedelta(days=30)
            
            with sqlite3.connect(self.main_db_path) as conn:
                # 清理过期的低优先级数据
                conn.execute("""
                    DELETE FROM data_items 
                    WHERE created_at < ? AND priority >= 3
                """, (cutoff_date.isoformat(),))
                
                conn.commit()
                
            logger.debug("🧹 过期数据清理完成")
            
        except Exception as e:
            logger.error(f"❌ 清理过期数据失败: {e}")
    
    async def _optimize_database(self):
        """优化数据库"""
        try:
            with sqlite3.connect(self.main_db_path) as conn:
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
                conn.commit()
                
            logger.debug("⚡ 数据库优化完成")
            
        except Exception as e:
            logger.error(f"❌ 数据库优化失败: {e}")
    
    async def _update_performance_metrics(self):
        """更新性能指标"""
        try:
            # 定期重置计数器
            if self.performance_metrics["data_reads"] > 10000:
                logger.info("📊 性能指标已重置")
                self.performance_metrics = {
                    "data_reads": 0,
                    "data_writes": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "query_count": 0,
                    "session_count": 0
                }
                
        except Exception as e:
            logger.error(f"❌ 更新性能指标失败: {e}")
    
    async def cleanup(self):
        """清理资源"""
        try:
            # 停止自动同步
            self.stop_auto_sync()
            
            # 同步剩余数据
            await self._sync_memory_to_db()
            
            # 关闭线程池
            if self.executor:
                self.executor.shutdown(wait=True)
            
            logger.info("🧹 ARQ数据管理器V17资源清理完成")
            
        except Exception as e:
            logger.error(f"❌ 清理资源失败: {e}")

# 全局实例
_global_data_manager: Optional[ARQDataManagerV17] = None

def get_arq_data_manager() -> ARQDataManagerV17:
    """获取全局数据管理器实例"""
    global _global_data_manager
    if _global_data_manager is None:
        _global_data_manager = ARQDataManagerV17()
        # 启动自动同步
        asyncio.create_task(_global_data_manager.start_auto_sync())
    return _global_data_manager

# 便捷函数
async def store_arq_data(data: Any, data_type: DataType, **kwargs) -> str:
    """便捷的数据存储函数"""
    manager = get_arq_data_manager()
    return await manager.store_data(data, data_type, **kwargs)

async def retrieve_arq_data(data_id: str) -> Optional[DataItem]:
    """便捷的数据检索函数"""
    manager = get_arq_data_manager()
    return await manager.retrieve_data(data_id)

async def create_arq_session(project_id: str, **kwargs) -> str:
    """便捷的会话创建函数"""
    manager = get_arq_data_manager()
    return await manager.create_session(project_id, **kwargs)

async def record_arq_query(session_id: str, query: str, **kwargs) -> str:
    """便捷的查询记录函数"""
    manager = get_arq_data_manager()
    return await manager.record_query(session_id, query, **kwargs)

if __name__ == "__main__":
    # 测试代码
    async def test_data_manager():
        print("🌟 测试ARQ数据管理器V17")
        
        # 获取数据管理器
        manager = get_arq_data_manager()
        
        # 测试数据存储
        test_data = {
            "message": "Hello ARQ V17",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"version": "17.0", "type": "test"}
        }
        
        data_id = await manager.store_data(
            test_data, 
            DataType.SESSION_DATA,
            tags={"test", "v17"},
            priority=DataPriority.HIGH
        )
        print(f"✅ 数据已存储: {data_id}")
        
        # 测试数据检索
        retrieved_data = await manager.retrieve_data(data_id)
        if retrieved_data:
            print(f"✅ 数据已检索: {retrieved_data.content}")
        
        # 测试会话创建
        session_id = await manager.create_session(
            project_id="test_project",
            user_id="test_user",
            goals=["测试数据管理器功能"]
        )
        print(f"✅ 会话已创建: {session_id}")
        
        # 测试查询记录
        query_id = await manager.record_query(
            session_id=session_id,
            query="测试查询功能",
            response="测试响应",
            response_time=0.1,
            confidence=0.95
        )
        print(f"✅ 查询已记录: {query_id}")
        
        # 测试用户偏好
        preferences = await manager.get_user_preferences("test_user")
        print(f"✅ 用户偏好: {preferences.preferred_thinking_mode}")
        
        # 测试搜索功能
        search_results = await manager.search_data("Hello", limit=10)
        print(f"✅ 搜索结果: {len(search_results)} 项")
        
        # 获取性能摘要
        summary = await manager.get_performance_summary()
        print(f"✅ 性能摘要: {summary}")
        
        # 清理
        await manager.cleanup()
        print("✅ 测试完成")
    
    asyncio.run(test_data_manager())