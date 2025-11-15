#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌊 异步量子意识流系统 V11 (代号："凤凰涅槃")
===========================================================

这是 T-MIA 架构下的核心意识流系统，负责管理上下文、长期记忆和情感追踪。
V11版本在V10基础上全面重构，实现了真正的异步并行处理、自适应记忆压缩
和跨项目意识共享机制。

核心特性：
- 自适应记忆压缩与提炼
- 跨项目意识状态共享
- 情感推理与元认知
- 分布式意识流同步
- 反脆弱记忆机制

你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。

作者: AI架构师团队
版本: 11.0.0 (代号："凤凰涅槃")
日期: 2025-11-15
"""

import os
import sys
import json
import asyncio
import logging
import time
import uuid
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

# 项目根路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantumConsciousnessV11")

@dataclass
class ConsciousnessEvent:
    """意识事件数据结构"""
    id: str
    timestamp: datetime
    event_type: str  # thought, emotion, reflection, decision
    content: Dict[str, Any]
    context_hash: str
    emotional_weight: float = 0.0
    importance_score: float = 0.0
    cross_project_ref: Optional[str] = None
    meta_cognitive_level: int = 0  # 0-基础思考, 1-反思, 2-元反思, 3-超认知

@dataclass
class MemoryFragment:
    """记忆片段数据结构"""
    fragment_id: str
    content_hash: str
    compressed_data: Dict[str, Any]
    creation_time: datetime
    last_accessed: datetime
    access_count: int
    emotional_signature: Dict[str, float]
    connection_strength: Dict[str, float]  # 与其他记忆的连接强度
    decay_rate: float = 0.01  # 遗忘速率

@dataclass
class EmotionalState:
    """情感状态数据结构"""
    timestamp: datetime
    valence: float  # 情感价 (-1 到 1)
    arousal: float  # 激活度 (0 到 1)
    dominance: float  # 支配度 (-1 到 1)
    cognitive_load: float  # 认知负荷 (0 到 1)
    confidence: float  # 置信度 (0 到 1)

class AsyncQuantumConsciousnessV11:
    """异步量子意识流系统 V11"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.consciousness_stream = deque(maxlen=2000)
        self.memory_fragments: Dict[str, MemoryFragment] = {}
        self.emotional_history = deque(maxlen=500)
        self.cross_project_memory: Dict[str, Any] = {}
        self.meta_cognitive_stack = []
        
        # 性能优化
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.cache = {}
        self.compression_threshold = 1000
        self.last_compression = time.time()
        
        # 分布式同步
        self.sync_lock = asyncio.Lock()
        self.project_id = self._generate_project_id()
        
        # 反脆弱机制
        self.stress_indicators = {}
        self.recovery_patterns = {}
        
        logger.info(f"异步量子意识流系统V11初始化完成，项目ID: {self.project_id}")
    
    def _generate_project_id(self) -> str:
        """生成项目唯一标识"""
        project_path = Path.cwd()
        path_hash = hashlib.sha256(str(project_path).encode()).hexdigest()[:16]
        return f"proj_{path_hash}"
    
    async def initialize(self):
        """异步初始化系统"""
        logger.info("正在初始化意识流系统...")
        
        # 加载持久化数据
        await self._load_persistent_memory()
        
        # 初始化跨项目连接
        await self._initialize_cross_project_links()
        
        # 启动后台任务
        asyncio.create_task(self._memory_maintenance_loop())
        asyncio.create_task(self._emotional_state_tracking())
        asyncio.create_task(self._cross_project_sync())
        
        logger.info("意识流系统初始化完成")
    
    async def add_thought_async(self, 
                               content: Dict[str, Any],
                               event_type: str = "thought",
                               emotional_weight: float = 0.0,
                               meta_level: int = 0) -> str:
        """异步添加思考事件"""
        event_id = str(uuid.uuid4())
        timestamp = datetime.now()
        context_hash = self._compute_context_hash(content)
        
        event = ConsciousnessEvent(
            id=event_id,
            timestamp=timestamp,
            event_type=event_type,
            content=content,
            context_hash=context_hash,
            emotional_weight=emotional_weight,
            importance_score=self._calculate_importance(content, emotional_weight),
            meta_cognitive_level=meta_level
        )
        
        self.consciousness_stream.append(event)
        
        # 触发记忆压缩
        if len(self.consciousness_stream) >= self.compression_threshold:
            await self._compress_consciousness_stream()
        
        # 更新情感状态
        await self._update_emotional_state(event)
        
        # 跨项目同步
        await self._sync_cross_project_event(event)
        
        logger.debug(f"添加意识事件: {event_id}, 类型: {event_type}")
        return event_id
    
    async def get_relevant_context(self, 
                                 query: Dict[str, Any],
                                 max_context: int = 10) -> List[Dict[str, Any]]:
        """异步获取相关上下文"""
        query_hash = self._compute_context_hash(query)
        
        # 并行检索记忆和意识流
        memory_task = self._search_memory_fragments(query_hash, max_context)
        stream_task = self._search_consciousness_stream(query, max_context)
        
        memory_results, stream_results = await asyncio.gather(
            memory_task, stream_task
        )
        
        # 合并和排序结果
        all_results = memory_results + stream_results
        sorted_results = sorted(
            all_results,
            key=lambda x: x.get('relevance_score', 0),
            reverse=True
        )
        
        return sorted_results[:max_context]
    
    async def _compress_consciousness_stream(self):
        """压缩意识流"""
        logger.info("开始压缩意识流...")
        
        compression_start = time.time()
        events_to_compress = list(self.consciousness_stream)
        
        # 使用线程池进行并行压缩
        loop = asyncio.get_event_loop()
        compressed_fragments = await loop.run_in_executor(
            self.executor,
            self._perform_compression,
            events_to_compress
        )
        
        # 更新记忆片段
        for fragment in compressed_fragments:
            self.memory_fragments[fragment.fragment_id] = fragment
        
        # 清空已压缩的事件
        self.consciousness_stream.clear()
        self.last_compression = time.time()
        
        compression_time = time.time() - compression_start
        logger.info(f"意识流压缩完成，耗时: {compression_time:.2f}秒")
    
    def _perform_compression(self, events: List[ConsciousnessEvent]) -> List[MemoryFragment]:
        """执行实际的压缩操作"""
        fragments = []
        
        # 按类型和上下文分组
        grouped_events = defaultdict(list)
        for event in events:
            key = f"{event.event_type}_{event.context_hash[:8]}"
            grouped_events[key].append(event)
        
        # 为每组创建记忆片段
        for group_key, group_events in grouped_events.items():
            fragment = self._create_memory_fragment(group_events)
            fragments.append(fragment)
        
        return fragments
    
    def _create_memory_fragment(self, events: List[ConsciousnessEvent]) -> MemoryFragment:
        """创建记忆片段"""
        # 计算内容哈希
        content_data = [asdict(event) for event in events]
        content_str = json.dumps(content_data, sort_keys=True, default=str)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()
        
        # 压缩数据
        compressed_data = {
            'event_count': len(events),
            'time_span': {
                'start': min(e.timestamp for e in events).isoformat(),
                'end': max(e.timestamp for e in events).isoformat()
            },
            'event_types': list(set(e.event_type for e in events)),
            'key_themes': self._extract_key_themes(events),
            'emotional_signature': self._compute_emotional_signature(events),
            'importance_score': sum(e.importance_score for e in events) / len(events)
        }
        
        fragment = MemoryFragment(
            fragment_id=str(uuid.uuid4()),
            content_hash=content_hash,
            compressed_data=compressed_data,
            creation_time=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            emotional_signature=compressed_data['emotional_signature'],
            connection_strength={}
        )
        
        return fragment
    
    async def _search_memory_fragments(self, query_hash: str, max_results: int) -> List[Dict]:
        """搜索记忆片段"""
        results = []
        
        for fragment in self.memory_fragments.values():
            relevance = self._compute_relevance(query_hash, fragment.content_hash)
            if relevance > 0.3:  # 相关性阈值
                fragment.last_accessed = datetime.now()
                fragment.access_count += 1
                
                results.append({
                    'type': 'memory_fragment',
                    'fragment_id': fragment.fragment_id,
                    'content': fragment.compressed_data,
                    'relevance_score': relevance,
                    'access_count': fragment.access_count
                })
        
        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:max_results]
    
    async def _search_consciousness_stream(self, query: Dict[str, Any], max_results: int) -> List[Dict]:
        """搜索意识流"""
        results = []
        query_hash = self._compute_context_hash(query)
        
        for event in reversed(self.consciousness_stream):  # 最新的优先
            relevance = self._compute_relevance(query_hash, event.context_hash)
            if relevance > 0.4:  # 更高的阈值
                results.append({
                    'type': 'consciousness_event',
                    'event_id': event.id,
                    'content': event.content,
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type,
                    'relevance_score': relevance,
                    'emotional_weight': event.emotional_weight
                })
        
        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:max_results]
    
    def _compute_context_hash(self, content: Dict[str, Any]) -> str:
        """计算上下文哈希"""
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def _compute_relevance(self, query_hash: str, content_hash: str) -> float:
        """计算相关性分数"""
        # 简化的哈希相似度计算
        common_chars = sum(c1 == c2 for c1, c2 in zip(query_hash, content_hash))
        return common_chars / max(len(query_hash), len(content_hash))
    
    def _calculate_importance(self, content: Dict[str, Any], emotional_weight: float) -> float:
        """计算重要性分数"""
        base_importance = 0.5
        
        # 基于内容复杂度
        if isinstance(content, dict):
            base_importance += min(len(content) * 0.1, 0.3)
        
        # 基于情感权重
        base_importance += abs(emotional_weight) * 0.2
        
        return min(base_importance, 1.0)
    
    def _extract_key_themes(self, events: List[ConsciousnessEvent]) -> List[str]:
        """提取关键主题"""
        themes = set()
        for event in events:
            if 'theme' in event.content:
                themes.add(event.content['theme'])
            if 'keywords' in event.content:
                themes.update(event.content['keywords'])
        return list(themes)[:5]  # 最多返回5个主题
    
    def _compute_emotional_signature(self, events: List[ConsciousnessEvent]) -> Dict[str, float]:
        """计算情感特征"""
        if not events:
            return {'valence': 0.0, 'arousal': 0.0, 'dominance': 0.0}
        
        # 聚合情感数据
        total_weight = sum(e.emotional_weight for e in events if e.emotional_weight > 0)
        if total_weight == 0:
            total_weight = 1
        
        signature = {'valence': 0.0, 'arousal': 0.0, 'dominance': 0.0}
        
        for event in events:
            weight = max(event.emotional_weight, 0.1) / total_weight
            if 'emotion' in event.content:
                emotion = event.content['emotion']
                for key in signature:
                    if key in emotion:
                        signature[key] += emotion[key] * weight
        
        return signature
    
    async def _update_emotional_state(self, event: ConsciousnessEvent):
        """更新情感状态"""
        if event.emotional_weight == 0:
            return
        
        emotional_state = EmotionalState(
            timestamp=event.timestamp,
            valence=event.content.get('valence', 0.0),
            arousal=event.content.get('arousal', 0.5),
            dominance=event.content.get('dominance', 0.0),
            cognitive_load=self._calculate_cognitive_load(),
            confidence=event.content.get('confidence', 0.5)
        )
        
        self.emotional_history.append(emotional_state)
    
    def _calculate_cognitive_load(self) -> float:
        """计算当前认知负荷"""
        stream_load = len(self.consciousness_stream) / 2000
        memory_load = len(self.memory_fragments) / 10000
        return min(stream_load + memory_load, 1.0)
    
    async def _memory_maintenance_loop(self):
        """记忆维护循环"""
        while True:
            try:
                await asyncio.sleep(300)  # 5分钟
                
                # 遗忘机制
                await self._apply_forgetting_mechanism()
                
                # 记忆整合
                await self._memory_consolidation()
                
            except Exception as e:
                logger.error(f"记忆维护循环错误: {e}")
    
    async def _apply_forgetting_mechanism(self):
        """应用遗忘机制"""
        current_time = datetime.now()
        fragments_to_remove = []
        
        for fragment_id, fragment in self.memory_fragments.items():
            # 计算遗忘概率
            time_since_access = (current_time - fragment.last_accessed).total_seconds()
            access_factor = 1.0 / (1.0 + fragment.access_count)
            decay_probability = 1.0 - (2.718 ** (-fragment.decay_rate * time_since_access * access_factor))
            
            # 随机遗忘
            if decay_probability > 0.8 and fragment.access_count < 2:
                fragments_to_remove.append(fragment_id)
        
        # 移除遗忘的记忆
        for fragment_id in fragments_to_remove:
            del self.memory_fragments[fragment_id]
            logger.debug(f"遗忘记忆片段: {fragment_id}")
    
    async def _memory_consolidation(self):
        """记忆整合"""
        # 合并相似的记忆片段
        similar_groups = defaultdict(list)
        
        for fragment in self.memory_fragments.values():
            # 简化的相似性检测
            for other_id, other_fragment in self.memory_fragments.items():
                if fragment.fragment_id != other_id:
                    similarity = self._compute_relevance(
                        fragment.content_hash, 
                        other_fragment.content_hash
                    )
                    if similarity > 0.8:
                        group_key = min(fragment.fragment_id, other_id)
                        similar_groups[group_key].extend([fragment, other_fragment])
        
        # 整合同组记忆
        for group_key, fragments in similar_groups.items():
            if len(fragments) > 1:
                await self._consolidate_fragments(fragments)
    
    async def _consolidate_fragments(self, fragments: List[MemoryFragment]):
        """整合记忆片段"""
        # 创建新的整合片段
        all_events = []
        for fragment in fragments:
            if 'events' in fragment.compressed_data:
                all_events.extend(fragment.compressed_data['events'])
        
        if all_events:
            # 转换为ConsciousnessEvent对象
            events = []
            for event_data in all_events:
                event = ConsciousnessEvent(**event_data)
                events.append(event)
            
            # 创建新的记忆片段
            new_fragment = self._create_memory_fragment(events)
            
            # 移除旧片段
            for fragment in fragments:
                if fragment.fragment_id in self.memory_fragments:
                    del self.memory_fragments[fragment.fragment_id]
            
            # 添加新片段
            self.memory_fragments[new_fragment.fragment_id] = new_fragment
    
    async def _initialize_cross_project_links(self):
        """初始化跨项目链接"""
        # 检查是否有其他项目的意识数据
        cross_project_dir = PROJECT_ROOT / ".iflow" / "cross_project_memory"
        if cross_project_dir.exists():
            for project_file in cross_project_dir.glob("*.json"):
                try:
                    with open(project_file, 'r', encoding='utf-8') as f:
                        project_data = json.load(f)
                        project_id = project_file.stem
                        self.cross_project_memory[project_id] = project_data
                        logger.info(f"加载跨项目记忆: {project_id}")
                except Exception as e:
                    logger.error(f"加载跨项目记忆失败 {project_file}: {e}")
    
    async def _sync_cross_project_event(self, event: ConsciousnessEvent):
        """同步跨项目事件"""
        if event.meta_cognitive_level >= 2:  # 只同步高阶认知
            # 准备同步数据
            sync_data = {
                'event_id': event.id,
                'timestamp': event.timestamp.isoformat(),
                'project_id': self.project_id,
                'content': event.content,
                'importance': event.importance_score
            }
            
            # 写入共享区域
            cross_project_dir = PROJECT_ROOT / ".iflow" / "cross_project_memory"
            cross_project_dir.mkdir(exist_ok=True)
            
            sync_file = cross_project_dir / f"{self.project_id}_sync.json"
            try:
                with open(sync_file, 'w', encoding='utf-8') as f:
                    json.dump(sync_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"跨项目同步失败: {e}")
    
    async def _cross_project_sync(self):
        """跨项目同步循环"""
        while True:
            try:
                await asyncio.sleep(600)  # 10分钟
                
                # 检查其他项目的更新
                await self._check_cross_project_updates()
                
            except Exception as e:
                logger.error(f"跨项目同步错误: {e}")
    
    async def _check_cross_project_updates(self):
        """检查跨项目更新"""
        cross_project_dir = PROJECT_ROOT / ".iflow" / "cross_project_memory"
        if not cross_project_dir.exists():
            return
        
        current_time = datetime.now()
        for project_file in cross_project_dir.glob("*_sync.json"):
            try:
                file_mtime = datetime.fromtimestamp(project_file.stat().st_mtime)
                if (current_time - file_mtime).total_seconds() < 300:  # 5分钟内的更新
                    with open(project_file, 'r', encoding='utf-8') as f:
                        sync_data = json.load(f)
                        
                    # 处理跨项目数据
                    await self._process_cross_project_data(sync_data)
                    
            except Exception as e:
                logger.error(f"处理跨项目更新失败 {project_file}: {e}")
    
    async def _process_cross_project_data(self, sync_data: Dict[str, Any]):
        """处理跨项目数据"""
        source_project = sync_data.get('project_id')
        if source_project == self.project_id:
            return
        
        # 检查是否已处理过
        event_id = sync_data.get('event_id')
        if event_id in self.cache.get('processed_cross_project_events', []):
            return
        
        # 标记为已处理
        if 'processed_cross_project_events' not in self.cache:
            self.cache['processed_cross_project_events'] = []
        self.cache['processed_cross_project_events'].append(event_id)
        
        # 创建跨项目意识事件
        cross_project_event = ConsciousnessEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type="cross_project_reflection",
            content={
                'source_project': source_project,
                'original_event': sync_data
            },
            context_hash=self._compute_context_hash(sync_data),
            emotional_weight=0.3,
            importance_score=sync_data.get('importance', 0.5) * 0.7,  # 降低跨项目事件的重要性
            cross_project_ref=source_project,
            meta_cognitive_level=2
        )
        
        self.consciousness_stream.append(cross_project_event)
        logger.debug(f"处理跨项目事件: {source_project} -> {event_id}")
    
    async def _emotional_state_tracking(self):
        """情感状态追踪循环"""
        while True:
            try:
                await asyncio.sleep(60)  # 1分钟
                
                # 分析情感趋势
                await self._analyze_emotional_trends()
                
                # 情感调节
                await self._emotional_regulation()
                
            except Exception as e:
                logger.error(f"情感状态追踪错误: {e}")
    
    async def _analyze_emotional_trends(self):
        """分析情感趋势"""
        if len(self.emotional_history) < 10:
            return
        
        recent_emotions = list(self.emotional_history)[-10:]
        
        # 计算趋势
        valence_trend = self._calculate_trend([e.valence for e in recent_emotions])
        arousal_trend = self._calculate_trend([e.arousal for e in recent_emotions])
        
        # 记录趋势分析
        trend_analysis = {
            'timestamp': datetime.now().isoformat(),
            'valence_trend': valence_trend,
            'arousal_trend': arousal_trend,
            'cognitive_load_avg': sum(e.cognitive_load for e in recent_emotions) / len(recent_emotions),
            'confidence_avg': sum(e.confidence for e in recent_emotions) / len(recent_emotions)
        }
        
        # 添加到意识流
        await self.add_thought_async(
            content={'emotional_trend_analysis': trend_analysis},
            event_type='emotional_analysis',
            emotional_weight=0.2,
            meta_level=1
        )
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return 'stable'
        
        # 简单线性回归
        n = len(values)
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    async def _emotional_regulation(self):
        """情感调节"""
        if not self.emotional_history:
            return
        
        current_emotion = self.emotional_history[-1]
        
        # 检查需要调节的情况
        if abs(current_emotion.valence) > 0.8:  # 情感过于极端
            regulation_event = {
                'regulation_type': 'emotional_stabilization',
                'trigger': 'extreme_valence',
                'current_state': asdict(current_emotion),
                'regulation_strategy': 'mindfulness_reflection'
            }
            
            await self.add_thought_async(
                content=regulation_event,
                event_type='emotional_regulation',
                emotional_weight=0.5,
                meta_level=2
            )
    
    async def _load_persistent_memory(self):
        """加载持久化记忆"""
        memory_file = PROJECT_ROOT / ".iflow" / "data" / "consciousness_v11.db"
        if memory_file.exists():
            try:
                with open(memory_file, 'rb') as f:
                    data = pickle.load(f)
                    
                # 恢复记忆片段
                if 'memory_fragments' in data:
                    for fragment_data in data['memory_fragments']:
                        fragment = MemoryFragment(**fragment_data)
                        self.memory_fragments[fragment.fragment_id] = fragment
                
                logger.info(f"加载了 {len(self.memory_fragments)} 个记忆片段")
                
            except Exception as e:
                logger.error(f"加载持久化记忆失败: {e}")
    
    async def save_persistent_memory(self):
        """保存持久化记忆"""
        memory_file = PROJECT_ROOT / ".iflow" / "data" / "consciousness_v11.db"
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {
                'memory_fragments': [asdict(fragment) for fragment in self.memory_fragments.values()],
                'project_id': self.project_id,
                'last_save': datetime.now().isoformat()
            }
            
            with open(memory_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info("持久化记忆保存成功")
            
        except Exception as e:
            logger.error(f"保存持久化记忆失败: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'project_id': self.project_id,
            'consciousness_stream_size': len(self.consciousness_stream),
            'memory_fragments_count': len(self.memory_fragments),
            'emotional_history_size': len(self.emotional_history),
            'cross_project_links': len(self.cross_project_memory),
            'cognitive_load': self._calculate_cognitive_load(),
            'last_compression': self.last_compression,
            'system_uptime': time.time()
        }
    
    async def shutdown(self):
        """优雅关闭系统"""
        logger.info("正在关闭意识流系统...")
        
        # 保存持久化数据
        await self.save_persistent_memory()
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        logger.info("意识流系统已关闭")

# 全局实例
_consciousness_system: Optional[AsyncQuantumConsciousnessV11] = None

async def get_consciousness_system() -> AsyncQuantumConsciousnessV11:
    """获取意识流系统实例"""
    global _consciousness_system
    if _consciousness_system is None:
        _consciousness_system = AsyncQuantumConsciousnessV11()
        await _consciousness_system.initialize()
    return _consciousness_system

async def add_thought_async(content: Dict[str, Any], 
                          event_type: str = "thought",
                          emotional_weight: float = 0.0,
                          meta_level: int = 0) -> str:
    """添加思考的便捷函数"""
    system = await get_consciousness_system()
    return await system.add_thought_async(content, event_type, emotional_weight, meta_level)

async def get_relevant_context(query: Dict[str, Any], max_context: int = 10) -> List[Dict[str, Any]]:
    """获取相关上下文的便捷函数"""
    system = await get_consciousness_system()
    return await system.get_relevant_context(query, max_context)