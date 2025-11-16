#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 智能提示词优化器 V1.0 (Intelligent Prompt Optimizer)
====================================================

为ARP系统添加智能提示词优化功能：
- 🎯 自动优化用户提示词
- 👤 用户画像和偏好学习
- 🔤 5种优化模式（标准/专业/小白/AI格式/重新优化）
- 💾 本地数据持久化存储
- 📈 自动训练和持续学习
- 🌊 断点式交互优化
- 🎨 个性化适配
- 🚀 越用越懂用户

核心功能：
1. 提示词智能优化
2. 用户画像构建
3. 偏好学习系统
4. 本地数据管理
5. 自动训练机制
6. 多模式适配

作者: iFlow架构团队
版本: 1.0.0
日期: 2025-11-17
"""

import os
import sys
import json
import asyncio
import logging
import time
import uuid
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

# 项目根路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationMode(Enum):
    """优化模式枚举"""
    STANDARD = "standard"           # 标准优化
    PROFESSIONAL = "professional"   # 专业方向
    BEGINNER = "beginner"          # 小白听得懂
    AI_FORMAT = "ai_format"       # AI听得懂格式
    REOPTIMIZE = "reoptimize"     # 重新优化

class UserExpertiseLevel(Enum):
    """用户专业水平"""
    EXPERT = "expert"      # 专家
    ADVANCED = "advanced"  # 高级
    INTERMEDIATE = "intermediate"  # 中级
    BEGINNER = "beginner"  # 初学者

@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    name: Optional[str] = None
    expertise_level: UserExpertiseLevel = UserExpertiseLevel.INTERMEDIATE
    preferred_language: str = "zh"
    preferred_complexity: str = "balanced"  # simple, balanced, complex
    interaction_style: str = "direct"  # direct, detailed, casual
    field_of_interest: List[str] = field(default_factory=list)
    optimization_preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    total_interactions: int = 0
    satisfaction_scores: List[float] = field(default_factory=list)

@dataclass
class PromptOptimizationRecord:
    """提示词优化记录"""
    record_id: str
    user_id: str
    original_prompt: str
    optimized_prompt: str
    optimization_mode: OptimizationMode
    user_feedback: Optional[int] = None  # 1-5分
    user_accepted: bool = False
    optimization_reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0

@dataclass
class OptimizationResult:
    """优化结果"""
    success: bool
    optimized_prompt: str
    optimization_mode: OptimizationMode
    reasoning: str
    confidence: float
    suggestions: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)

class IntelligentPromptOptimizer:
    """智能提示词优化器"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or PROJECT_ROOT / "data" / "prompt_optimizer"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储路径
        self.profiles_file = self.data_dir / "user_profiles.json"
        self.history_file = self.data_dir / "optimization_history.json"
        self.training_data_file = self.data_dir / "training_data.json"
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # 内存数据
        self.user_profiles: Dict[str, UserProfile] = {}
        self.optimization_history: List[PromptOptimizationRecord] = []
        self.training_data: List[Dict[str, Any]] = []
        
        # 优化规则和模板
        self.optimization_rules = self._load_optimization_rules()
        self.mode_templates = self._load_mode_templates()
        
        # 加载现有数据
        self._load_data()
        
        # 线程锁
        self._lock = threading.Lock()
        
        logger.info("🧠 智能提示词优化器初始化完成")
    
    def _load_optimization_rules(self) -> Dict[str, List[str]]:
        """加载优化规则"""
        return {
            "clarity": [
                "使用明确、具体的语言",
                "避免模糊和歧义的表达",
                "确保逻辑结构清晰"
            ],
            "completeness": [
                "包含必要的上下文信息",
                "明确期望的输出格式",
                "提供相关的约束条件"
            ],
            "effectiveness": [
                "使用行动导向的动词",
                "合理设置优先级",
                "提供示例和模板"
            ],
            "efficiency": [
                "去除冗余信息",
                "精简表达方式",
                "优化提示词结构"
            ]
        }
    
    def _load_mode_templates(self) -> Dict[OptimizationMode, Dict[str, Any]]:
        """加载模式模板"""
        return {
            OptimizationMode.STANDARD: {
                "description": "标准优化模式，平衡清晰度和完整性",
                "focus_areas": ["clarity", "completeness", "effectiveness"],
                "style": "balanced",
                "complexity": "medium"
            },
            OptimizationMode.PROFESSIONAL: {
                "description": "专业方向优化，使用行业术语和专业表达",
                "focus_areas": ["completeness", "effectiveness"],
                "style": "formal",
                "complexity": "high",
                "additions": ["专业术语", "技术细节", "行业标准"]
            },
            OptimizationMode.BEGINNER: {
                "description": "小白友好模式，简单易懂的表达",
                "focus_areas": ["clarity", "simplicity"],
                "style": "casual",
                "complexity": "low",
                "additions": ["简单解释", "步骤说明", "通俗比喻"]
            },
            OptimizationMode.AI_FORMAT: {
                "description": "AI友好格式，结构化提示词",
                "focus_areas": ["structure", "precision"],
                "style": "structured",
                "complexity": "medium",
                "additions": ["结构化格式", "明确指令", "角色定义"]
            },
            OptimizationMode.REOPTIMIZE: {
                "description": "重新优化，基于反馈改进",
                "focus_areas": ["all"],
                "style": "adaptive",
                "complexity": "variable",
                "additions": ["反馈整合", "问题修复", "性能提升"]
            }
        }
    
    def _load_data(self):
        """加载持久化数据"""
        try:
            # 加载用户画像
            if self.profiles_file.exists():
                with open(self.profiles_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for user_id, profile_data in data.items():
                        profile = UserProfile(**profile_data)
                        # 转换日期字符串
                        if isinstance(profile.created_at, str):
                            profile.created_at = datetime.fromisoformat(profile.created_at)
                        if isinstance(profile.last_updated, str):
                            profile.last_updated = datetime.fromisoformat(profile.last_updated)
                        self.user_profiles[user_id] = profile
            
            # 加载优化历史
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for record_data in data:
                        record = PromptOptimizationRecord(**record_data)
                        if isinstance(record.timestamp, str):
                            record.timestamp = datetime.fromisoformat(record.timestamp)
                        self.optimization_history.append(record)
            
            # 加载训练数据
            if self.training_data_file.exists():
                with open(self.training_data_file, 'r', encoding='utf-8') as f:
                    self.training_data = json.load(f)
            
            logger.info(f"✅ 数据加载完成: {len(self.user_profiles)}个用户, {len(self.optimization_history)}条历史")
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
    
    def _save_data(self):
        """保存数据到文件"""
        try:
            with self._lock:
                # 保存用户画像
                profiles_data = {}
                for user_id, profile in self.user_profiles.items():
                    profile_dict = asdict(profile)
                    profile_dict['created_at'] = profile.created_at.isoformat()
                    profile_dict['last_updated'] = profile.last_updated.isoformat()
                    # 转换枚举为字符串
                    profile_dict['expertise_level'] = profile.expertise_level.value
                    profiles_data[user_id] = profile_dict
                
                with open(self.profiles_file, 'w', encoding='utf-8') as f:
                    json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                
                # 保存优化历史
                history_data = []
                for record in self.optimization_history:
                    record_dict = asdict(record)
                    record_dict['timestamp'] = record.timestamp.isoformat()
                    # 转换枚举为字符串
                    record_dict['optimization_mode'] = record.optimization_mode.value
                    history_data.append(record_dict)
                
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump(history_data, f, ensure_ascii=False, indent=2)
                
                # 保存训练数据
                with open(self.training_data_file, 'w', encoding='utf-8') as f:
                    json.dump(self.training_data, f, ensure_ascii=False, indent=2)
                
                logger.debug("💾 数据保存完成")
                
        except Exception as e:
            logger.error(f"❌ 数据保存失败: {e}")
    
    def get_or_create_user(self, user_id: str, name: Optional[str] = None) -> UserProfile:
        """获取或创建用户画像"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                name=name or f"用户_{user_id[:8]}"
            )
            self._save_data()
        
        return self.user_profiles[user_id]
    
    def update_user_profile(self, user_id: str, **kwargs):
        """更新用户画像"""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            for key, value in kwargs.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            profile.last_updated = datetime.now()
            self._save_data()
    
    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """分析提示词特征"""
        analysis = {
            "length": len(prompt),
            "word_count": len(prompt.split()),
            "sentence_count": len(re.split(r'[。！？.!?]+', prompt)),
            "has_context": bool(re.search(r'背景|上下文|场景|情况', prompt)),
            "has_constraints": bool(re.search(r'限制|要求|必须|避免', prompt)),
            "has_examples": bool(re.search(r'例如|比如|示例|例子', prompt)),
            "has_format": bool(re.search(r'格式|结构|模板|样式', prompt)),
            "clarity_score": 0.0,
            "completeness_score": 0.0,
            "complexity": "medium"
        }
        
        # 计算清晰度分数
        clarity_indicators = [
            analysis['has_context'],
            not len(prompt) < 10,
            analysis['word_count'] > 3,
            '?' not in prompt or prompt.count('?') <= 2
        ]
        analysis['clarity_score'] = sum(clarity_indicators) / len(clarity_indicators)
        
        # 计算完整性分数
        completeness_indicators = [
            analysis['has_context'],
            analysis['has_constraints'],
            analysis['has_examples'] or analysis['has_format']
        ]
        analysis['completeness_score'] = sum(completeness_indicators) / len(completeness_indicators)
        
        # 判断复杂度
        if analysis['word_count'] < 20:
            analysis['complexity'] = "low"
        elif analysis['word_count'] > 100:
            analysis['complexity'] = "high"
        
        return analysis
    
    def _optimize_for_mode(self, prompt: str, mode: OptimizationMode, user_profile: UserProfile) -> Tuple[str, str]:
        """根据模式优化提示词"""
        analysis = self._analyze_prompt(prompt)
        template = self.mode_templates[mode]
        
        optimized = prompt
        reasoning_steps = []
        
        # 基础优化
        if "clarity" in template.get("focus_areas", []):
            if analysis['clarity_score'] < 0.7:
                optimized = self._improve_clarity(optimized)
                reasoning_steps.append("提升表达清晰度")
        
        if "completeness" in template.get("focus_areas", []):
            if analysis['completeness_score'] < 0.7:
                optimized = self._improve_completeness(optimized)
                reasoning_steps.append("补充必要信息")
        
        # 模式特定优化
        if mode == OptimizationMode.PROFESSIONAL:
            optimized = self._add_professional_elements(optimized, user_profile)
            reasoning_steps.append("添加专业术语和技术细节")
        
        elif mode == OptimizationMode.BEGINNER:
            optimized = self._simplify_for_beginner(optimized)
            reasoning_steps.append("简化表达，增加解释")
        
        elif mode == OptimizationMode.AI_FORMAT:
            optimized = self._structure_for_ai(optimized)
            reasoning_steps.append("结构化格式，明确指令")
        
        elif mode == OptimizationMode.REOPTIMIZE:
            optimized = self._apply_feedback_learning(optimized, user_profile)
            reasoning_steps.append("基于历史反馈优化")
        
        # 通用优化
        optimized = self._general_optimization(optimized)
        if not reasoning_steps:
            reasoning_steps.append("通用优化改进")
        
        reasoning = f"优化步骤: {' → '.join(reasoning_steps)}"
        return optimized, reasoning
    
    def _improve_clarity(self, prompt: str) -> str:
        """提升清晰度"""
        # 添加明确的目标
        if not any(word in prompt for word in ['请', '帮我', '需要', '要求']):
            prompt = f"请{prompt}"
        
        # 去除模糊表达
        replacements = {
            '一些': '具体的',
            '可能': '确定',
            '大概': '准确',
            '左右': '精确'
        }
        
        for old, new in replacements.items():
            prompt = prompt.replace(old, new)
        
        return prompt
    
    def _improve_completeness(self, prompt: str) -> str:
        """提升完整性"""
        # 添加上下文要求
        if not any(word in prompt for word in ['背景', '上下文', '场景']):
            prompt += "\n请提供相关背景信息。"
        
        # 添加输出格式要求
        if not any(word in prompt for word in ['格式', '结构', '输出']):
            prompt += "\n请明确输出格式。"
        
        return prompt
    
    def _add_professional_elements(self, prompt: str, user_profile: UserProfile) -> str:
        """添加专业元素"""
        # 根据用户兴趣领域添加专业术语
        if user_profile.field_of_interest:
            field = user_profile.field_of_interest[0]  # 使用主要兴趣领域
            professional_additions = {
                "技术": ["技术实现", "架构设计", "性能优化"],
                "商业": ["商业价值", "市场分析", "ROI"],
                "学术": ["研究方法", "理论基础", "实验设计"],
                "艺术": ["创意理念", "美学原则", "表现形式"]
            }
            
            if field in professional_additions:
                additions = professional_additions[field]
                prompt += f"\n请从{', '.join(additions)}角度进行分析。"
        
        return prompt
    
    def _simplify_for_beginner(self, prompt: str) -> str:
        """为初学者简化"""
        # 添加解释性要求
        prompt += "\n请用简单易懂的语言解释，避免使用专业术语。"
        prompt += "\n如果需要，可以使用生活中的例子来说明。"
        
        return prompt
    
    def _structure_for_ai(self, prompt: str) -> str:
        """为AI结构化"""
        # 添加角色定义
        if not prompt.startswith("你是") and "角色" not in prompt:
            prompt = f"你是一个专业的助手。\n{prompt}"
        
        # 添加任务结构
        structured_prompt = f"""
## 任务目标
{prompt}

## 输出要求
1. 逻辑清晰，层次分明
2. 内容完整，重点突出
3. 格式规范，易于理解

## 约束条件
- 确保准确性
- 保持客观性
- 提供可操作性建议
"""
        return structured_prompt.strip()
    
    def _apply_feedback_learning(self, prompt: str, user_profile: UserProfile) -> str:
        """应用反馈学习"""
        # 获取用户历史优化记录
        user_history = [r for r in self.optimization_history if r.user_id == user_profile.user_id]
        
        if user_history:
            # 分析用户偏好
            accepted_modes = [r.optimization_mode.value for r in user_history if r.user_accepted]
            high_feedback = [r for r in user_history if r.user_feedback and r.user_feedback >= 4]
            
            if accepted_modes:
                # 应用用户偏好的模式
                preferred_mode = max(set(accepted_modes), key=accepted_modes.count)
                if preferred_mode != OptimizationMode.REOPTIMIZE.value:
                    prompt, _ = self._optimize_for_mode(prompt, OptimizationMode(preferred_mode), user_profile)
            
            if high_feedback:
                # 学习高分反馈的特征
                for record in high_feedback[-3:]:  # 最近3条高分记录
                    prompt = self._apply_successful_patterns(prompt, record.optimized_prompt)
        
        return prompt
    
    def _apply_successful_patterns(self, current_prompt: str, successful_prompt: str) -> str:
        """应用成功模式"""
        # 提取成功提示词的模式
        patterns = []
        
        # 检查结构模式
        if "##" in successful_prompt:
            patterns.append("structured_format")
        if "1." in successful_prompt:
            patterns.append("numbered_list")
        if "：" in successful_prompt and "，" in successful_prompt:
            patterns.append("detailed_explanation")
        
        # 应用模式
        if "structured_format" in patterns and "##" not in current_prompt:
            current_prompt = f"## 任务\n{current_prompt}"
        
        return current_prompt
    
    def _general_optimization(self, prompt: str) -> str:
        """通用优化"""
        # 去除多余空白
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        
        # 确保标点符号规范
        prompt = prompt.replace('，，', '，').replace('。。', '。')
        
        # 确保结尾有标点
        if prompt and prompt[-1] not in '。！？.!?':
            prompt += '。'
        
        return prompt
    
    async def optimize_prompt(self, user_id: str, original_prompt: str, mode: OptimizationMode = OptimizationMode.STANDARD) -> OptimizationResult:
        """优化提示词"""
        start_time = time.time()
        
        try:
            # 获取用户画像
            user_profile = self.get_or_create_user(user_id)
            
            # 分析原始提示词
            analysis = self._analyze_prompt(original_prompt)
            
            # 执行优化
            optimized_prompt, reasoning = self._optimize_for_mode(original_prompt, mode, user_profile)
            
            # 计算置信度
            confidence = self._calculate_confidence(analysis, optimized_prompt)
            
            # 生成建议
            suggestions = self._generate_suggestions(analysis, mode)
            
            # 生成下一步操作
            next_steps = [
                "输入 1: 确认使用优化后的提示词",
                "输入 2: 重新优化当前提示词",
                "输入 3: 切换到专业方向优化",
                "输入 4: 切换到小白友好模式",
                "输入 5: 切换到AI友好格式"
            ]
            
            # 创建优化记录
            record = PromptOptimizationRecord(
                record_id=str(uuid.uuid4()),
                user_id=user_id,
                original_prompt=original_prompt,
                optimized_prompt=optimized_prompt,
                optimization_mode=mode,
                optimization_reasoning=reasoning,
                processing_time=time.time() - start_time
            )
            
            # 保存记录
            self.optimization_history.append(record)
            self._save_data()
            
            # 更新用户交互次数
            user_profile.total_interactions += 1
            self._save_data()
            
            return OptimizationResult(
                success=True,
                optimized_prompt=optimized_prompt,
                optimization_mode=mode,
                reasoning=reasoning,
                confidence=confidence,
                suggestions=suggestions,
                next_steps=next_steps
            )
            
        except Exception as e:
            logger.error(f"❌ 提示词优化失败: {e}")
            return OptimizationResult(
                success=False,
                optimized_prompt=original_prompt,
                optimization_mode=mode,
                reasoning=f"优化失败: {str(e)}",
                confidence=0.0
            )
    
    def _calculate_confidence(self, analysis: Dict[str, Any], optimized_prompt: str) -> float:
        """计算优化置信度"""
        base_confidence = 0.7
        
        # 基于改进程度调整
        length_improvement = min(0.1, (len(optimized_prompt) - analysis['length']) / 100)
        clarity_bonus = (1 - analysis['clarity_score']) * 0.2
        completeness_bonus = (1 - analysis['completeness_score']) * 0.2
        
        confidence = base_confidence + length_improvement + clarity_bonus + completeness_bonus
        return min(1.0, max(0.0, confidence))
    
    def _generate_suggestions(self, analysis: Dict[str, Any], mode: OptimizationMode) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        if analysis['clarity_score'] < 0.7:
            suggestions.append("建议进一步明确表达意图")
        
        if analysis['completeness_score'] < 0.7:
            suggestions.append("建议添加更多上下文信息")
        
        if analysis['word_count'] < 10:
            suggestions.append("提示词可能过于简单，建议补充细节")
        
        if analysis['word_count'] > 200:
            suggestions.append("提示词较长，考虑简化表达")
        
        # 模式特定建议
        if mode == OptimizationMode.PROFESSIONAL:
            suggestions.append("专业模式已应用，确保符合行业标准")
        elif mode == OptimizationMode.BEGINNER:
            suggestions.append("已简化表达，适合初学者理解")
        
        return suggestions
    
    def record_feedback(self, record_id: str, user_feedback: int, user_accepted: bool):
        """记录用户反馈"""
        for record in self.optimization_history:
            if record.record_id == record_id:
                record.user_feedback = user_feedback
                record.user_accepted = user_accepted
                
                # 更新用户画像
                user_profile = self.user_profiles.get(record.user_id)
                if user_profile:
                    user_profile.satisfaction_scores.append(user_feedback)
                    user_profile.last_updated = datetime.now()
                    
                    # 自动调整用户专业水平
                    if user_feedback >= 4 and record.optimization_mode == OptimizationMode.PROFESSIONAL:
                        if user_profile.expertise_level == UserExpertiseLevel.BEGINNER:
                            user_profile.expertise_level = UserExpertiseLevel.INTERMEDIATE
                        elif user_profile.expertise_level == UserExpertiseLevel.INTERMEDIATE:
                            user_profile.expertise_level = UserExpertiseLevel.ADVANCED
                
                # 添加到训练数据
                self.training_data.append({
                    "original_prompt": record.original_prompt,
                    "optimized_prompt": record.optimized_prompt,
                    "mode": record.optimization_mode.value,
                    "feedback": user_feedback,
                    "accepted": user_accepted,
                    "timestamp": datetime.now().isoformat()
                })
                
                self._save_data()
                break
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户统计信息"""
        user_profile = self.get_or_create_user(user_id)
        user_history = [r for r in self.optimization_history if r.user_id == user_id]
        
        if not user_history:
            return {
                "total_interactions": 0,
                "acceptance_rate": 0.0,
                "average_satisfaction": 0.0,
                "preferred_modes": [],
                "expertise_level": user_profile.expertise_level.value
            }
        
        accepted_count = sum(1 for r in user_history if r.user_accepted)
        feedback_scores = [r.user_feedback for r in user_history if r.user_feedback is not None]
        
        mode_counts = {}
        for record in user_history:
            if isinstance(record.optimization_mode, str):
                mode = record.optimization_mode
            else:
                mode = record.optimization_mode.value
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        # 处理专业水平枚举
        expertise_level = user_profile.expertise_level.value if hasattr(user_profile.expertise_level, 'value') else user_profile.expertise_level
        
        return {
            "total_interactions": len(user_history),
            "acceptance_rate": accepted_count / len(user_history) * 100,
            "average_satisfaction": sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0.0,
            "preferred_modes": sorted(mode_counts.items(), key=lambda x: x[1], reverse=True),
            "expertise_level": expertise_level,
            "satisfaction_trend": user_profile.satisfaction_scores[-10:]  # 最近10次
        }
    
    def export_user_data(self, user_id: str, export_path: Optional[Path] = None) -> str:
        """导出用户数据"""
        if export_path is None:
            export_path = self.data_dir / f"user_data_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        user_profile = self.get_or_create_user(user_id)
        user_history = [r for r in self.optimization_history if r.user_id == user_id]
        user_stats = self.get_user_statistics(user_id)
        
        export_data = {
            "user_profile": asdict(user_profile),
            "optimization_history": [asdict(r) for r in user_history],
            "statistics": user_stats,
            "export_timestamp": datetime.now().isoformat()
        }
        
        # 转换日期格式
        export_data["user_profile"]["created_at"] = user_profile.created_at.isoformat()
        export_data["user_profile"]["last_updated"] = user_profile.last_updated.isoformat()
        
        for record in export_data["optimization_history"]:
            record["timestamp"] = record["timestamp"].isoformat()
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return str(export_path)
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """清理旧数据"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # 清理优化历史
        original_count = len(self.optimization_history)
        self.optimization_history = [
            r for r in self.optimization_history 
            if r.timestamp > cutoff_date
        ]
        
        # 清理训练数据
        original_training_count = len(self.training_data)
        self.training_data = [
            d for d in self.training_data
            if datetime.fromisoformat(d["timestamp"]) > cutoff_date
        ]
        
        self._save_data()
        
        logger.info(f"🧹 数据清理完成: 删除 {original_count - len(self.optimization_history)} 条历史记录, "
                   f"{original_training_count - len(self.training_data)} 条训练数据")

# 全局优化器实例
_global_optimizer: Optional[IntelligentPromptOptimizer] = None

def get_prompt_optimizer() -> IntelligentPromptOptimizer:
    """获取全局提示词优化器实例"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = IntelligentPromptOptimizer()
    return _global_optimizer

# 便捷函数
async def optimize_user_prompt(user_id: str, prompt: str, mode: str = "standard") -> OptimizationResult:
    """便捷的提示词优化函数"""
    optimizer = get_prompt_optimizer()
    try:
        optimization_mode = OptimizationMode(mode)
    except ValueError:
        optimization_mode = OptimizationMode.STANDARD
    
    return await optimizer.optimize_prompt(user_id, prompt, optimization_mode)

if __name__ == "__main__":
    # 测试智能提示词优化器
    async def test_optimizer():
        print("🧪 测试智能提示词优化器")
        
        optimizer = IntelligentPromptOptimizer()
        
        # 测试优化
        user_id = "test_user_001"
        test_prompt = "帮我写个代码"
        
        result = await optimizer.optimize_prompt(user_id, test_prompt, OptimizationMode.STANDARD)
        
        print(f"✅ 优化结果:")
        print(f"原始提示词: {test_prompt}")
        print(f"优化后: {result.optimized_prompt}")
        print(f"优化模式: {result.optimization_mode.value}")
        print(f"置信度: {result.confidence:.2f}")
        print(f"建议: {result.suggestions}")
        print(f"下一步: {result.next_steps}")
        
        # 测试反馈
        optimizer.record_feedback(
            record_id=optimizer.optimization_history[-1].record_id,
            user_feedback=5,
            user_accepted=True
        )
        
        # 查看统计
        stats = optimizer.get_user_statistics(user_id)
        print(f"📊 用户统计: {stats}")
        
        print("🎉 测试完成")
    
    asyncio.run(test_optimizer())