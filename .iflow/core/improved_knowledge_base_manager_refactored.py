#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📚 改进版本地知识库管理系统 V3.0 (重构版)
================================

重构版本，专注于：
- 低复杂度和高可维护性
- 单一职责原则
- 依赖注入和设计模式应用
- 模块化和可测试性

作者: AI架构师团队
版本: 3.0.0
日期: 2025-11-16
"""

import asyncio
import gc
import json
import logging
import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union,
    Protocol, runtime_checkable
)

import faiss
import numpy as np
import psutil

# 项目配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_BASE_ROOT = PROJECT_ROOT / "knowledge_base"

# 类型定义
T = TypeVar('T')


class iFlowException(Exception):
    """iFlow基础异常类"""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now()


class ConfigurationError(iFlowException):
    """配置错误"""
    pass


class ValidationError(iFlowException):
    """验证错误"""
    pass


class SearchError(iFlowException):
    """搜索错误"""
    pass


class ComponentError(iFlowException):
    """组件错误"""
    pass


class ComponentStatus(Enum):
    """组件状态"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class EventType(Enum):
    """事件类型"""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    COMPONENT_ADDED = "component_added"
    DOCUMENT_ADDED = "document_added"
    SEARCH_PERFORMED = "search_performed"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class BaseConfig:
    """基础配置类"""
    name: str
    description: str = ""
    version: str = "1.0.0"
    enabled: bool = True
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        if not self.name or not self.name.strip():
            errors.append("名称不能为空")
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class KnowledgeBaseConfig(BaseConfig):
    """知识库配置"""
    path: str = ""
    file_types: List[str] = field(
        default_factory=lambda: [".txt", ".md", ".pdf", ".docx", ".doc", ".html", ".py", ".js", ".json", ".xml"]
    )
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    auto_index: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_batch_size: int = 100
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_memory_usage_mb: int = 512
    cache_size: int = 1000
    index_cache_ttl: int = 3600
    auto_cleanup_interval: int = 300
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = super().validate()
        
        if not self.path:
            errors.append("路径不能为空")
        elif not Path(self.path).exists():
            errors.append(f"路径不存在: {self.path}")
        
        if self.max_file_size <= 0:
            errors.append("最大文件大小必须大于0")
        
        if self.chunk_size <= 0:
            errors.append("块大小必须大于0")
        
        if self.max_memory_usage_mb <= 0:
            errors.append("最大内存使用量必须大于0")
        
        return errors


@dataclass
class DocumentChunk:
    """文档块"""
    chunk_id: str
    doc_id: str
    group_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        """初始化后处理"""
        if not self.chunk_id:
            raise ValidationError("块ID不能为空")
        if not self.content:
            raise ValidationError("内容不能为空")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        if self.embedding is not None:
            result['embedding'] = self.embedding.tolist()
        return result


@dataclass
class KnowledgeGroup:
    """知识库组"""
    group_id: str
    name: str
    description: str
    path: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    document_count: int = 0
    total_size: int = 0
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """初始化后处理"""
        if not self.group_id:
            raise ValidationError("组ID不能为空")
        if not self.name:
            raise ValidationError("组名不能为空")
    
    def update_timestamp(self) -> None:
        """更新时间戳"""
        self.updated_at = datetime.now()
    
    def add_document(self, file_size: int) -> None:
        """添加文档"""
        self.document_count += 1
        self.total_size += file_size
        self.update_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        return result


@dataclass
class SearchResult:
    """搜索结果"""
    chunk_id: str
    doc_id: str
    group_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    highlights: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """初始化后处理"""
        if not 0 <= self.score <= 1:
            raise ValidationError("分数必须在0-1之间")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


# ============================================================================
# 接口定义
# ============================================================================

@runtime_checkable
class EventListener(Protocol):
    """事件监听器协议"""
    
    def handle_event(self, event_type: EventType, data: Any) -> None:
        """处理事件"""
        ...


class IConfigValidator(ABC):
    """配置验证器接口"""
    
    @abstractmethod
    def validate(self, config: BaseConfig) -> List[str]:
        """验证配置"""
        pass


class IIndexManager(ABC):
    """索引管理器接口"""
    
    @abstractmethod
    def initialize_index(self, embedding_dimension: int) -> None:
        """初始化索引"""
        pass
    
    @abstractmethod
    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """添加嵌入向量"""
        pass
    
    @abstractmethod
    def save_index(self) -> None:
        """保存索引"""
        pass


class IDocumentProcessor(ABC):
    """文档处理器接口"""
    
    @abstractmethod
    def create_chunks(self, content: str, doc_id: str, group_id: str, file_path: str) -> List[DocumentChunk]:
        """创建文档块"""
        pass


class ISearchStrategy(ABC):
    """搜索策略接口"""
    
    @abstractmethod
    def search(self, query: str, data: List[Any], top_k: int = 10) -> List[SearchResult]:
        """执行搜索"""
        pass


# ============================================================================
# 核心组件实现
# ============================================================================

class ConfigValidator(IConfigValidator):
    """配置验证器"""
    
    def validate(self, config: BaseConfig) -> List[str]:
        """验证配置"""
        return config.validate()


class EventDispatcher:
    """事件分发器"""
    
    def __init__(self) -> None:
        self._listeners: Dict[EventType, List[EventListener]] = defaultdict(list)
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)
    
    def add_listener(self, event_type: EventType, listener: EventListener) -> None:
        """添加事件监听器"""
        with self._lock:
            self._listeners[event_type].append(listener)
    
    def remove_listener(self, event_type: EventType, listener: EventListener) -> None:
        """移除事件监听器"""
        with self._lock:
            if listener in self._listeners[event_type]:
                self._listeners[event_type].remove(listener)
    
    def emit_event(self, event_type: EventType, data: Any = None) -> None:
        """发布事件"""
        with self._lock:
            listeners = self._listeners[event_type].copy()
        
        for listener in listeners:
            try:
                listener.handle_event(event_type, data)
            except Exception as e:
                self._logger.error(f"事件处理失败: {e}")


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
    
    def handle_error(self, error: Exception, context: str = "") -> None:
        """处理错误"""
        error_msg = f"{context}: {error}" if context else str(error)
        self._logger.error(error_msg)
        
        # 可以添加更多错误处理逻辑，如发送通知、记录到数据库等
        if isinstance(error, (ConfigurationError, ValidationError)):
            self._logger.warning(f"配置/验证错误: {error_msg}")
        elif isinstance(error, (SearchError, ComponentError)):
            self._logger.error(f"业务错误: {error_msg}")
        else:
            self._logger.critical(f"未知错误: {error_msg}")


class MemoryManager:
    """内存管理器"""
    
    def __init__(self, max_memory_mb: int, cleanup_interval: int = 300) -> None:
        self.max_memory_mb = max_memory_mb
        self.cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._logger = logging.getLogger(__name__)
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self) -> None:
        """启动内存监控"""
        if self._monitor_thread is None:
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            self._logger.info("内存监控已启动")
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while True:
            try:
                if self._check_memory_limit():
                    self._logger.warning(f"内存使用超限: {self._get_memory_usage():.2f}MB")
                    self._cleanup()
                    gc.collect()
                
                # 定期清理
                if time.time() - self._last_cleanup > self.cleanup_interval:
                    self._cleanup()
                    self._last_cleanup = time.time()
                
                time.sleep(60)  # 每分钟检查一次
            except Exception as e:
                self._logger.error(f"内存监控错误: {e}")
                time.sleep(60)
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _check_memory_limit(self) -> bool:
        """检查是否超过内存限制"""
        return self._get_memory_usage() > self.max_memory_mb
    
    def _cleanup(self) -> None:
        """清理内存"""
        gc.collect()
        self._logger.debug("执行内存清理")


class IndexManager(IIndexManager):
    """索引管理器"""
    
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.index: Optional[faiss.Index] = None
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)
    
    def initialize_index(self, embedding_dimension: int) -> None:
        """初始化索引"""
        with self._lock:
            try:
                if self.index_path.exists():
                    self.index = faiss.read_index(str(self.index_path))
                    self._logger.info(f"📖 加载现有索引，包含 {self.index.ntotal} 个向量")
                else:
                    self.index = faiss.IndexFlatIP(embedding_dimension)
                    self._logger.info("🆕 创建新的Faiss索引")
            except Exception as e:
                self._logger.error(f"索引初始化失败: {e}")
                self.index = faiss.IndexFlatIP(embedding_dimension)
    
    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """添加嵌入向量"""
        with self._lock:
            if self.index is None:
                raise ComponentError("索引未初始化")
            
            # 归一化嵌入向量
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms
            
            self.index.add(normalized_embeddings)
            self._logger.info(f"添加了 {len(embeddings)} 个向量到索引")
    
    def save_index(self) -> None:
        """保存索引"""
        if self.index is None:
            return
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            faiss.write_index(self.index, str(self.index_path))
            self._logger.info(f"💾 保存索引，包含 {self.index.ntotal} 个向量")
        except Exception as e:
            self._logger.error(f"保存索引失败: {e}")
    
    def get_index_size(self) -> int:
        """获取索引大小"""
        return self.index.ntotal if self.index else 0


class DocumentProcessor(IDocumentProcessor):
    """文档处理器"""
    
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._logger = logging.getLogger(__name__)
    
    def create_chunks(self, content: str, doc_id: str, group_id: str, file_path: str) -> List[DocumentChunk]:
        """创建文档块"""
        if not content.strip():
            self._logger.warning(f"文档内容为空: {file_path}")
            return []
        
        chunks = []
        words = content.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_content = " ".join(chunk_words)
            
            chunk = DocumentChunk(
                chunk_id=f"chunk_{doc_id}_{i}",
                doc_id=doc_id,
                group_id=group_id,
                content=chunk_content,
                metadata={
                    "file_path": file_path,
                    "chunk_index": i,
                    "word_count": len(chunk_words),
                    "char_count": len(chunk_content)
                }
            )
            
            chunks.append(chunk)
        
        self._logger.debug(f"为文档 {doc_id} 创建了 {len(chunks)} 个块")
        return chunks


class KeywordSearchStrategy(ISearchStrategy):
    """关键词搜索策略"""
    
    def search(self, query: str, data: List[Any], top_k: int = 10) -> List[SearchResult]:
        """执行关键词搜索"""
        query_words = set(query.lower().split())
        results = []
        
        for item in data:
            content = item.get('content', '').lower()
            content_words = set(content.split())
            
            # 计算关键词匹配度
            match_count = len(query_words & content_words)
            if match_count > 0:
                score = match_count / len(query_words)
                result = SearchResult(
                    chunk_id=item.get('chunk_id', ''),
                    doc_id=item.get('doc_id', ''),
                    group_id=item.get('group_id', ''),
                    content=item.get('content', ''),
                    score=score,
                    metadata=item.get('metadata', {}),
                    highlights=self._extract_highlights(content, query_words)
                )
                results.append(result)
        
        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _extract_highlights(self, content: str, query_words: Set[str]) -> List[str]:
        """提取高亮片段"""
        highlights = []
        for word in query_words:
            if word in content:
                start = content.find(word)
                if start != -1:
                    context_start = max(0, start - 20)
                    context_end = min(len(content), start + len(word) + 20)
                    highlight = content[context_start:context_end]
                    if highlight not in highlights:
                        highlights.append(highlight)
        return highlights


class VectorSearchStrategy(ISearchStrategy):
    """向量搜索策略"""
    
    def __init__(self, embedding_dimension: int = 384):
        self.embedding_dimension = embedding_dimension
    
    def search(self, query: str, data: List[Any], top_k: int = 10) -> List[SearchResult]:
        """执行向量搜索"""
        # 简化的向量搜索实现
        results = []
        for item in data[:top_k]:
            result = SearchResult(
                chunk_id=item.get('chunk_id', ''),
                doc_id=item.get('doc_id', ''),
                group_id=item.get('group_id', ''),
                content=item.get('content', ''),
                score=np.random.random(),  # 简化的分数计算
                metadata=item.get('metadata', {})
            )
            results.append(result)
        return results


class SearchEngine:
    """搜索引擎"""
    
    def __init__(self, default_strategy: ISearchStrategy) -> None:
        self._strategy = default_strategy
        self._logger = logging.getLogger(__name__)
    
    def set_strategy(self, strategy: ISearchStrategy) -> None:
        """设置搜索策略"""
        self._strategy = strategy
    
    def search(self, query: str, data: List[Any], top_k: int = 10) -> List[SearchResult]:
        """执行搜索"""
        if not query or not query.strip():
            raise ValidationError("查询不能为空")
        
        if top_k <= 0:
            raise ValidationError("top_k必须大于0")
        
        try:
            results = self._strategy.search(query, data, top_k)
            self._logger.info(f"🔍 搜索完成: 查询='{query}'，返回 {len(results)} 个结果")
            return results
        except Exception as e:
            raise SearchError(f"搜索失败: {e}")


# ============================================================================
# 依赖注入容器
# ============================================================================

class DIContainer:
    """依赖注入容器"""
    
    def __init__(self) -> None:
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
    
    def register(self, name: str, factory: Callable[[], Any], singleton: bool = True) -> None:
        """注册服务"""
        self._services[name] = (factory, singleton)
    
    def get(self, name: str) -> Any:
        """获取服务"""
        if name not in self._services:
            raise ValueError(f"服务未注册: {name}")
        
        factory, singleton = self._services[name]
        
        if singleton:
            if name not in self._singletons:
                self._singletons[name] = factory()
            return self._singletons[name]
        else:
            return factory()
    
    def has(self, name: str) -> bool:
        """检查服务是否存在"""
        return name in self._services


# ============================================================================
# 重构后的主管理器
# ============================================================================

class KnowledgeBaseManager:
    """重构版知识库管理器"""
    
    def __init__(self, config: Optional[KnowledgeBaseConfig] = None) -> None:
        self.config = config or self._create_default_config()
        self._logger = self._setup_logging()
        
        # 初始化依赖注入容器
        self._container = self._setup_container()
        
        # 获取依赖
        self._config_validator = self._container.get("config_validator")
        self._error_handler = self._container.get("error_handler")
        self._event_dispatcher = self._container.get("event_dispatcher")
        self._memory_manager = self._container.get("memory_manager")
        self._index_manager = self._container.get("index_manager")
        self._document_processor = self._container.get("document_processor")
        self._search_engine = self._container.get("search_engine")
        
        # 数据存储
        self.groups: Dict[str, KnowledgeGroup] = {}
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.chunks: List[DocumentChunk] = []
        
        # 初始化系统
        self._initialize()
    
    def _create_default_config(self) -> KnowledgeBaseConfig:
        """创建默认配置"""
        return KnowledgeBaseConfig(
            name="default",
            description="默认知识库",
            path=str(KNOWLEDGE_BASE_ROOT / "documents")
        )
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        log_dir = KNOWLEDGE_BASE_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "knowledge_manager.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _setup_container(self) -> DIContainer:
        """设置依赖注入容器"""
        container = DIContainer()
        
        # 注册服务
        container.register("config_validator", lambda: ConfigValidator())
        container.register("error_handler", lambda: ErrorHandler(self._logger))
        container.register("event_dispatcher", lambda: EventDispatcher())
        container.register(
            "memory_manager", 
            lambda: MemoryManager(self.config.max_memory_usage_mb, self.config.auto_cleanup_interval)
        )
        container.register(
            "index_manager",
            lambda: IndexManager(KNOWLEDGE_BASE_ROOT / "indexes" / "faiss_index.bin")
        )
        container.register(
            "document_processor",
            lambda: DocumentProcessor(self.config.chunk_size, self.config.chunk_overlap)
        )
        container.register(
            "search_engine",
            lambda: SearchEngine(VectorSearchStrategy())
        )
        
        return container
    
    def _initialize(self) -> None:
        """初始化系统"""
        try:
            # 验证配置
            errors = self._config_validator.validate(self.config)
            if errors:
                raise ConfigurationError(f"配置验证失败: {', '.join(errors)}")
            
            # 创建必要的目录
            self._create_directories()
            
            # 加载数据
            self._load_groups()
            
            # 初始化索引
            self._index_manager.initialize_index(384)
            
            # 启动后台服务
            self._memory_manager.start_monitoring()
            
            # 发送启动事件
            self._event_dispatcher.emit_event(EventType.SYSTEM_START)
            self._logger.info("📚 知识库管理器初始化完成")
            
        except Exception as e:
            self._error_handler.handle_error(e, "初始化失败")
            raise ComponentError(f"初始化失败: {e}")
    
    def _create_directories(self) -> None:
        """创建必要的目录"""
        directories = ["groups", "indexes", "logs", "config", "cache"]
        for dir_name in directories:
            (KNOWLEDGE_BASE_ROOT / dir_name).mkdir(exist_ok=True)
    
    def _load_groups(self) -> None:
        """加载组信息"""
        groups_file = KNOWLEDGE_BASE_ROOT / "config" / "groups.json"
        if groups_file.exists():
            try:
                with open(groups_file, 'r', encoding='utf-8') as f:
                    groups_data = json.load(f)
                    for group_data in groups_data:
                        group = KnowledgeGroup(
                            group_id=group_data["group_id"],
                            name=group_data["name"],
                            description=group_data["description"],
                            path=group_data["path"],
                            created_at=datetime.fromisoformat(group_data["created_at"]),
                            updated_at=datetime.fromisoformat(group_data["updated_at"]),
                            document_count=group_data.get("document_count", 0),
                            total_size=group_data.get("total_size", 0),
                            tags=group_data.get("tags", [])
                        )
                        self.groups[group.group_id] = group
            except Exception as e:
                self._error_handler.handle_error(e, "加载组信息失败")
    
    def _save_groups(self) -> None:
        """保存组信息"""
        groups_file = KNOWLEDGE_BASE_ROOT / "config" / "groups.json"
        try:
            groups_data = [group.to_dict() for group in self.groups.values()]
            with open(groups_file, 'w', encoding='utf-8') as f:
                json.dump(groups_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._error_handler.handle_error(e, "保存组信息失败")
    
    def create_group(
        self, 
        name: str, 
        description: str, 
        path: str, 
        tags: Optional[List[str]] = None
    ) -> str:
        """创建知识库组"""
        if not name or not name.strip():
            raise ValidationError("组名不能为空")
        
        if not path:
            raise ValidationError("路径不能为空")
        
        try:
            group_id = f"group_{int(time.time() * 1000)}"
            
            # 创建目录
            group_path = Path(path)
            group_path.mkdir(parents=True, exist_ok=True)
            
            # 创建组对象
            group = KnowledgeGroup(
                group_id=group_id,
                name=name.strip(),
                description=description.strip(),
                path=str(group_path),
                tags=tags or []
            )
            
            self.groups[group_id] = group
            self._save_groups()
            
            self._event_dispatcher.emit_event(
                EventType.COMPONENT_ADDED, 
                {"type": "group", "id": group_id, "name": name}
            )
            
            self._logger.info(f"✅ 创建知识库组: {name} (ID: {group_id})")
            return group_id
            
        except Exception as e:
            self._error_handler.handle_error(e, f"创建组失败: {name}")
            raise ComponentError(f"创建组失败: {e}")
    
    def add_documents_from_path(
        self, 
        path: str, 
        group_id: str, 
        recursive: bool = True
    ) -> Dict[str, Any]:
        """从路径添加文档"""
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"路径不存在: {path}")
        
        if group_id not in self.groups:
            raise ValidationError(f"组ID不存在: {group_id}")
        
        group = self.groups[group_id]
        added_files = []
        errors = []
        
        try:
            # 查找支持的文件
            pattern = "**/*" if recursive else "*"
            
            for file_path in path_obj.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in self.config.file_types:
                    try:
                        # 检查文件大小
                        if file_path.stat().st_size > self.config.max_file_size:
                            errors.append(f"{file_path.name}: 文件过大")
                            continue
                        
                        # 读取文件内容
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        
                        # 创建文档和块
                        doc_id = f"doc_{int(time.time() * 1000)}_{len(self.documents)}"
                        chunks = self._document_processor.create_chunks(content, doc_id, group_id, str(file_path))
                        
                        # 保存文档信息
                        self.documents[doc_id] = {
                            "doc_id": doc_id,
                            "file_path": str(file_path),
                            "group_id": group_id,
                            "file_name": file_path.name,
                            "file_size": file_path.stat().st_size,
                            "created_at": datetime.now().isoformat(),
                            "chunks": [chunk.chunk_id for chunk in chunks]
                        }
                        
                        self.chunks.extend(chunks)
                        added_files.append(str(file_path))
                        
                        # 更新组统计
                        group.add_document(file_path.stat().st_size)
                        
                    except Exception as e:
                        errors.append(f"{file_path.name}: {str(e)}")
            
            # 保存更新
            self._save_groups()
            self._update_index()
            
            self._event_dispatcher.emit_event(
                EventType.DOCUMENT_ADDED,
                {"count": len(added_files), "group_id": group_id}
            )
            
            result = {
                "added_files": added_files,
                "errors": errors,
                "total_chunks": len([c for c in self.chunks if c.group_id == group_id])
            }
            
            self._logger.info(f"📄 添加文档完成: {len(added_files)} 成功, {len(errors)} 失败")
            return result
            
        except Exception as e:
            self._error_handler.handle_error(e, "添加文档失败")
            raise ComponentError(f"添加文档失败: {e}")
    
    def _update_index(self) -> None:
        """更新索引"""
        if not self.chunks:
            return
        
        # 生成嵌入向量（简化版）
        embeddings = []
        for chunk in self.chunks[-self.config.index_batch_size:]:
            embedding = np.random.rand(384).astype('float32')
            embeddings.append(embedding)
            chunk.embedding = embedding
        
        if embeddings:
            embeddings_array = np.vstack(embeddings)
            self._index_manager.add_embeddings(embeddings_array)
            self._index_manager.save_index()
    
    def set_search_strategy(self, strategy: ISearchStrategy) -> None:
        """设置搜索策略"""
        self._search_engine.set_strategy(strategy)
    
    async def search(
        self, 
        query: str, 
        top_k: int = 10, 
        group_id: Optional[str] = None
    ) -> List[SearchResult]:
        """搜索知识库"""
        search_start = time.time()
        
        try:
            # 准备搜索数据
            search_data = []
            for chunk in self.chunks:
                if group_id and chunk.group_id != group_id:
                    continue
                search_data.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "group_id": chunk.group_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata
                })
            
            # 执行搜索
            results = self._search_engine.search(query, search_data, top_k)
            
            # 记录搜索事件
            self._event_dispatcher.emit_event(
                EventType.SEARCH_PERFORMED,
                {
                    "query": query,
                    "top_k": top_k,
                    "group_id": group_id,
                    "results_count": len(results),
                    "search_time": time.time() - search_start
                }
            )
            
            return results
            
        except Exception as e:
            self._error_handler.handle_error(e, "搜索失败")
            raise
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        total_docs = len(self.documents)
        total_chunks = len(self.chunks)
        total_size = sum(doc.get("file_size", 0) for doc in self.documents.values())
        
        group_stats = []
        for group in self.groups.values():
            group_stats.append(group.to_dict())
        
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_size_mb": total_size / (1024 * 1024),
            "total_groups": len(self.groups),
            "index_size": self._index_manager.get_index_size(),
            "memory_usage_mb": self._memory_manager._get_memory_usage(),
            "groups": group_stats,
            "last_updated": datetime.now().isoformat()
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_knowledge_base_manager(config: Optional[KnowledgeBaseConfig] = None) -> KnowledgeBaseManager:
    """创建知识库管理器实例"""
    return KnowledgeBaseManager(config)


# ============================================================================
# 测试函数
# ============================================================================

async def test_refactored_knowledge_base() -> None:
    """测试重构版知识库"""
    try:
        # 创建管理器
        manager = create_knowledge_base_manager()
        
        # 创建测试组
        group_id = manager.create_group(
            name="重构测试知识库",
            description="用于重构验证的知识库",
            path=str(KNOWLEDGE_BASE_ROOT / "test_docs"),
            tags=["测试", "重构"]
        )
        
        # 创建测试文档
        test_doc_path = KNOWLEDGE_BASE_ROOT / "test_docs" / "test.txt"
        test_doc_path.parent.mkdir(exist_ok=True)
        test_doc_path.write_text(
            "这是重构版本的测试文档，用于验证重构效果。"
            "包含一些示例内容和测试数据。"
            "支持中文和英文混合搜索。",
            encoding='utf-8'
        )
        
        # 添加文档
        result = manager.add_documents_from_path(
            path=str(test_doc_path.parent),
            group_id=group_id
        )
        
        print("添加结果:", result)
        
        # 设置关键词搜索策略
        manager.set_search_strategy(KeywordSearchStrategy())
        
        # 搜索测试
        search_results = await manager.search("测试", top_k=5)
        
        print("\n搜索结果:")
        for result in search_results:
            print(f"- {result.doc_id}: {result.score:.3f}")
            print(f"  内容: {result.content[:100]}...")
            if result.highlights:
                print(f"  高亮: {result.highlights}")
        
        # 获取统计信息
        stats = manager.get_knowledge_base_stats()
        print("\n统计信息:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    asyncio.run(test_refactored_knowledge_base())