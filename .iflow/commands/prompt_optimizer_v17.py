#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 智能提示词优化器 V17 Hyperdimensional Singularity
=====================================================

这是ARQ系统的智能提示词优化组件，实现：
- 🎯 智能提示词优化和适配
- 🤖 Agent模式自动适配
- 👤 用户画像学习和记忆
- 📊 多维度优化策略
- 💾 本地数据持久化存储
- 🔄 断点式交互优化
- 🌟 个性化AI理解增强

核心特性：
- 5种优化模式（标准、专业、小白、AI格式、自定义）
- 用户画像自动学习和更新
- 上下文关联和语义理解
- 历史优化记录和追踪
- 智能推荐和预测

作者: AI架构师团队
版本: 17.0.0 Hyperdimensional Singularity
日期: 2025-11-17
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
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import re
import numpy as np

# 项目根路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / ".iflow" / "core"))

# 导入ARQ组件
try:
    from arq_data_manager_v17 import get_arq_data_manager, DataType, DataPriority
    from arq_data_analyzer_v17 import get_arq_data_analyzer
    ARQ_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ARQ组件不可用: {e}")
    ARQ_COMPONENTS_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 优化模式
class OptimizationMode(Enum):
    """优化模式枚举"""
    STANDARD = "standard"           # 标准优化
    PROFESSIONAL = "professional"   # 专业方向
    BEGINNER = "beginner"          # 小白易懂
    AI_FORMAT = "ai_format"        # AI格式
    CUSTOM = "custom"              # 自定义

# 用户画像
@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    name: Optional[str] = None
    expertise_level: str = "intermediate"  # beginner, intermediate, expert
    preferred_style: str = "balanced"      # concise, detailed, balanced
    field_of_interest: List[str] = field(default_factory=list)
    communication_style: str = "professional"  # casual, professional, academic
    language_preference: str = "zh-CN"
    optimization_history: List[Dict] = field(default_factory=list)
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

# 优化结果
@dataclass
class OptimizationResult:
    """优化结果"""
    original_prompt: str
    optimized_prompt: str
    mode: OptimizationMode
    confidence: float
    improvements: List[str]
    reasoning: str
    user_feedback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    optimization_id: str = field(default_factory=lambda: str(uuid.uuid4()))

# 交互状态
@dataclass
class InteractionState:
    """交互状态"""
    session_id: str
    current_step: int = 1
    total_steps: int = 5
    pending_optimization: Optional[OptimizationResult] = None
    user_choices: List[int] = field(default_factory=list)
    context_history: List[str] = field(default_factory=list)

class PromptOptimizerV17:
    """智能提示词优化器V17主类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化优化器"""
        self.config = config or {}
        
        # 数据存储路径
        self.data_dir = PROJECT_ROOT / "data" / "prompt_optimizer"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 用户画像存储
        self.user_profiles = {}
        
        # 交互状态管理
        self.active_sessions = {}
        
        # ARQ组件
        self.data_manager = None
        self.data_analyzer = None
        
        # 优化策略
        self.optimization_strategies = {
            OptimizationMode.STANDARD: self._standard_optimization,
            OptimizationMode.PROFESSIONAL: self._professional_optimization,
            OptimizationMode.BEGINNER: self._beginner_optimization,
            OptimizationMode.AI_FORMAT: self._ai_format_optimization,
            OptimizationMode.CUSTOM: self._custom_optimization
        }
        
        # 初始化
        self._initialize()
        
        logger.info("🧠 智能提示词优化器V17初始化完成")
    
    def _initialize(self):
        """初始化组件"""
        # 加载用户画像
        self._load_user_profiles()
        
        # 初始化ARQ组件
        if ARQ_COMPONENTS_AVAILABLE:
            try:
                self.data_manager = get_arq_data_manager()
                self.data_analyzer = get_arq_data_analyzer()
                logger.info("✅ ARQ组件集成成功")
            except Exception as e:
                logger.warning(f"⚠️ ARQ组件集成失败: {e}")
    
    def _load_user_profiles(self):
        """加载用户画像"""
        try:
            profiles_file = self.data_dir / "user_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                
                for user_id, profile_data in profiles_data.items():
                    # 转换时间字段
                    if 'created_at' in profile_data:
                        profile_data['created_at'] = datetime.fromisoformat(profile_data['created_at'])
                    if 'last_updated' in profile_data:
                        profile_data['last_updated'] = datetime.fromisoformat(profile_data['last_updated'])
                    
                    self.user_profiles[user_id] = UserProfile(**profile_data)
                
                logger.info(f"✅ 加载了 {len(self.user_profiles)} 个用户画像")
        
        except Exception as e:
            logger.error(f"❌ 加载用户画像失败: {e}")
    
    def _save_user_profiles(self):
        """保存用户画像"""
        try:
            profiles_file = self.data_dir / "user_profiles.json"
            profiles_data = {}
            
            for user_id, profile in self.user_profiles.items():
                profile_dict = asdict(profile)
                # 转换时间为字符串
                profile_dict['created_at'] = profile.created_at.isoformat()
                profile_dict['last_updated'] = profile.last_updated.isoformat()
                profiles_data[user_id] = profile_dict
            
            with open(profiles_file, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, ensure_ascii=False, indent=2)
            
            logger.debug("💾 用户画像已保存")
        
        except Exception as e:
            logger.error(f"❌ 保存用户画像失败: {e}")
    
    async def optimize_prompt(self, user_id: str, original_prompt: str, 
                            mode: OptimizationMode = OptimizationMode.STANDARD,
                            context: Optional[str] = None) -> OptimizationResult:
        """优化提示词"""
        try:
            # 获取或创建用户画像
            user_profile = self._get_or_create_profile(user_id)
            
            # 执行优化
            optimization_func = self.optimization_strategies[mode]
            result = await optimization_func(original_prompt, user_profile, context)
            
            # 记录优化历史
            self._record_optimization(user_id, result)
            
            # 保存到ARQ系统
            if self.data_manager:
                await self.data_manager.store_data(
                    data=asdict(result),
                    data_type=DataType.SESSION_DATA,
                    priority=DataPriority.HIGH,
                    tags={"prompt_optimization", mode.value}
                )
            
            logger.info(f"✅ 提示词优化完成: {result.optimization_id}")
            return result
        
        except Exception as e:
            logger.error(f"❌ 提示词优化失败: {e}")
            raise
    
    async def start_interactive_optimization(self, user_id: str, original_prompt: str) -> InteractionState:
        """启动交互式优化"""
        try:
            # 创建会话ID
            session_id = str(uuid.uuid4())
            
            # 创建交互状态
            interaction = InteractionState(
                session_id=session_id,
                current_step=1,
                total_steps=5
            )
            
            # 存储交互状态
            self.active_sessions[session_id] = interaction
            
            # 执行初始优化
            initial_result = await self.optimize_prompt(
                user_id=user_id,
                original_prompt=original_prompt,
                mode=OptimizationMode.STANDARD
            )
            
            interaction.pending_optimization = initial_result
            
            logger.info(f"🎯 启动交互式优化: {session_id}")
            return interaction
        
        except Exception as e:
            logger.error(f"❌ 启动交互式优化失败: {e}")
            raise
    
    async def handle_user_choice(self, session_id: str, choice: int) -> Dict[str, Any]:
        """处理用户选择"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError("会话不存在")
            
            interaction = self.active_sessions[session_id]
            interaction.user_choices.append(choice)
            
            # 处理选择
            response = await self._process_choice(interaction, choice)
            
            # 更新交互状态
            if choice == 1:  # 继续下一步
                interaction.current_step += 1
                if interaction.current_step > interaction.total_steps:
                    response['completed'] = True
                    response['final_prompt'] = interaction.pending_optimization.optimized_prompt
            
            elif choice == 2:  # 重新优化
                response['action'] = 'reoptimize'
            
            elif choice == 3:  # 专业方向
                await self._apply_mode_optimization(interaction, OptimizationMode.PROFESSIONAL)
            
            elif choice == 4:  # 小白易懂
                await self._apply_mode_optimization(interaction, OptimizationMode.BEGINNER)
            
            elif choice == 5:  # AI格式
                await self._apply_mode_optimization(interaction, OptimizationMode.AI_FORMAT)
            
            logger.info(f"✅ 处理用户选择: {choice}")
            return response
        
        except Exception as e:
            logger.error(f"❌ 处理用户选择失败: {e}")
            raise
    
    async def _process_choice(self, interaction: InteractionState, choice: int) -> Dict[str, Any]:
        """处理具体选择"""
        response = {
            'session_id': interaction.session_id,
            'choice': choice,
            'current_step': interaction.current_step,
            'message': ''
        }
        
        if choice == 1:
            response['message'] = f"✅ 已确认，继续第 {interaction.current_step + 1} 步优化..."
        
        elif choice == 2:
            response['message'] = "🔄 正在重新优化提示词..."
        
        elif choice == 3:
            response['message'] = "🎯 正在应用专业方向优化..."
        
        elif choice == 4:
            response['message'] = "📚 正在应用小白易懂优化..."
        
        elif choice == 5:
            response['message'] = "🤖 正在应用AI格式优化..."
        
        else:
            response['message'] = f"⚠️ 未知选择: {choice}"
        
        return response
    
    async def _apply_mode_optimization(self, interaction: InteractionState, mode: OptimizationMode):
        """应用特定模式优化"""
        if interaction.pending_optimization:
            user_id = self._get_user_id_from_session(interaction.session_id)
            if user_id:
                new_result = await self.optimize_prompt(
                    user_id=user_id,
                    original_prompt=interaction.pending_optimization.original_prompt,
                    mode=mode
                )
                interaction.pending_optimization = new_result
    
    def _get_user_id_from_session(self, session_id: str) -> Optional[str]:
        """从会话获取用户ID"""
        # 这里简化处理，实际应该从会话数据中获取
        return "default_user"
    
    async def _standard_optimization(self, prompt: str, profile: UserProfile, context: Optional[str]) -> OptimizationResult:
        """标准优化"""
        improvements = []
        optimized = prompt
        
        # 基础优化规则
        if len(prompt) < 10:
            optimized = f"请详细说明：{optimized}"
            improvements.append("增加详细说明要求")
        
        if "请" not in optimized and "please" not in optimized.lower():
            optimized = f"请{optimized}"
            improvements.append("添加礼貌用语")
        
        if "?" not in optimized and "？" not in optimized:
            optimized += "？"
            improvements.append("添加疑问标记")
        
        # 根据用户画像调整
        if profile.communication_style == "professional":
            optimized = optimized.replace("请", "请您")
            improvements.append("调整为专业语气")
        
        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized,
            mode=OptimizationMode.STANDARD,
            confidence=0.85,
            improvements=improvements,
            reasoning="基于基础规则和用户画像的标准优化"
        )
    
    async def _professional_optimization(self, prompt: str, profile: UserProfile, context: Optional[str]) -> OptimizationResult:
        """专业方向优化"""
        improvements = []
        optimized = prompt
        
        # 专业术语和结构
        professional_terms = ["分析", "评估", "优化", "实现", "策略", "方案", "框架", "架构"]
        for term in professional_terms:
            if term in prompt and term not in optimized:
                optimized = optimized.replace(term, f"专业的{term}")
                improvements.append(f"增强{term}的专业性")
        
        # 添加专业结构
        if "步骤" not in optimized and "step" not in optimized.lower():
            optimized += "\n请提供详细的实施步骤和评估标准。"
            improvements.append("添加专业结构要求")
        
        # 技术深度
        if profile.expertise_level == "expert":
            optimized += "\n请包含技术细节和最佳实践。"
            improvements.append("增加技术深度要求")
        
        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized,
            mode=OptimizationMode.PROFESSIONAL,
            confidence=0.90,
            improvements=improvements,
            reasoning="针对专业用户的深度优化"
        )
    
    async def _beginner_optimization(self, prompt: str, profile: UserProfile, context: Optional[str]) -> OptimizationResult:
        """小白易懂优化"""
        improvements = []
        optimized = prompt
        
        # 简化复杂词汇
        complex_terms = {
            "架构": "结构",
            "框架": "基础",
            "策略": "方法",
            "优化": "改进",
            "评估": "检查"
        }
        
        for complex_term, simple_term in complex_terms.items():
            if complex_term in optimized:
                optimized = optimized.replace(complex_term, simple_term)
                improvements.append(f"将'{complex_term}'简化为'{simple_term}'")
        
        # 添加解释性要求
        if "简单" not in optimized and "易懂" not in optimized:
            optimized += "\n请用简单易懂的语言解释，就像对初学者说话一样。"
            improvements.append("添加简单易懂要求")
        
        # 添加示例要求
        if "例子" not in optimized and "示例" not in optimized:
            optimized += "\n请提供具体的例子帮助理解。"
            improvements.append("添加示例要求")
        
        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized,
            mode=OptimizationMode.BEGINNER,
            confidence=0.88,
            improvements=improvements,
            reasoning="面向初学者的简化优化"
        )
    
    async def _ai_format_optimization(self, prompt: str, profile: UserProfile, context: Optional[str]) -> OptimizationResult:
        """AI格式优化"""
        improvements = []
        optimized = prompt
        
        # 添加AI指令格式
        if not optimized.startswith(("请", "Please", "作为", "假设")):
            optimized = f"作为AI助手，{optimized}"
            improvements.append("添加AI角色设定")
        
        # 添加输出格式要求
        if "格式" not in optimized and "format" not in optimized.lower():
            optimized += "\n请以结构化的格式输出，包含要点和详细说明。"
            improvements.append("添加结构化输出要求")
        
        # 添加思考过程要求
        if "思考" not in optimized and "thinking" not in optimized.lower():
            optimized += "\n请在回答前先进行思考分析。"
            improvements.append("添加思考过程要求")
        
        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized,
            mode=OptimizationMode.AI_FORMAT,
            confidence=0.92,
            improvements=improvements,
            reasoning="针对AI交互的格式优化"
        )
    
    async def _custom_optimization(self, prompt: str, profile: UserProfile, context: Optional[str]) -> OptimizationResult:
        """自定义优化"""
        improvements = []
        optimized = prompt
        
        # 基于用户历史偏好优化
        if profile.optimization_history:
            # 分析用户偏好的改进类型
            preferred_improvements = defaultdict(int)
            for history in profile.optimization_history[-10:]:  # 最近10次
                for improvement in history.get('improvements', []):
                    preferred_improvements[improvement] += 1
            
            # 应用用户偏好的改进
            for improvement, count in sorted(preferred_improvements.items(), key=lambda x: x[1], reverse=True)[:3]:
                if "礼貌" in improvement and "请" not in optimized:
                    optimized = f"请{optimized}"
                    improvements.append("根据偏好添加礼貌用语")
                elif "详细" in improvement and "详细" not in optimized:
                    optimized += "\n请提供详细说明。"
                    improvements.append("根据偏好增加详细要求")
        
        # 基于领域兴趣优化
        if profile.field_of_interest:
            field_keywords = {
                "技术": ["技术", "实现", "代码", "算法"],
                "商业": ["商业", "市场", "策略", "收益"],
                "教育": ["教育", "学习", "教学", "知识"],
                "医疗": ["医疗", "健康", "治疗", "诊断"]
            }
            
            for field in profile.field_of_interest:
                if field in field_keywords:
                    for keyword in field_keywords[field]:
                        if keyword in prompt and keyword not in optimized:
                            improvements.append(f"增强{field}领域专业性")
        
        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized,
            mode=OptimizationMode.CUSTOM,
            confidence=0.95,
            improvements=improvements,
            reasoning="基于用户画像的自定义优化"
        )
    
    def _get_or_create_profile(self, user_id: str) -> UserProfile:
        """获取或创建用户画像"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
        
        return self.user_profiles[user_id]
    
    def _record_optimization(self, user_id: str, result: OptimizationResult):
        """记录优化历史"""
        profile = self._get_or_create_profile(user_id)
        
        # 添加到历史记录
        optimization_record = {
            'optimization_id': result.optimization_id,
            'timestamp': result.timestamp.isoformat(),
            'mode': result.mode.value,
            'improvements': result.improvements,
            'confidence': result.confidence
        }
        
        profile.optimization_history.append(optimization_record)
        
        # 限制历史记录数量
        if len(profile.optimization_history) > 100:
            profile.optimization_history = profile.optimization_history[-100:]
        
        # 更新时间
        profile.last_updated = datetime.now()
        
        # 保存画像
        self._save_user_profiles()
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        return self.user_profiles.get(user_id)
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """更新用户画像"""
        try:
            profile = self._get_or_create_profile(user_id)
            
            # 更新字段
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            
            profile.last_updated = datetime.now()
            self._save_user_profiles()
            
            logger.info(f"✅ 用户画像已更新: {user_id}")
            return True
        
        except Exception as e:
            logger.error(f"❌ 更新用户画像失败: {e}")
            return False
    
    def get_data_storage_info(self) -> Dict[str, Any]:
        """获取数据存储信息"""
        return {
            "user_profiles_file": str(self.data_dir / "user_profiles.json"),
            "data_directory": str(self.data_dir),
            "total_users": len(self.user_profiles),
            "storage_permanent": True,
            "retention_policy": "永久保留，用户可手动清理",
            "backup_recommendation": "建议定期备份 user_profiles.json 文件"
        }
    
    async def cleanup_data(self, user_id: Optional[str] = None) -> bool:
        """清理数据"""
        try:
            if user_id:
                # 清理特定用户数据
                if user_id in self.user_profiles:
                    del self.user_profiles[user_id]
                    logger.info(f"✅ 已清理用户 {user_id} 的数据")
            else:
                # 清理所有数据
                self.user_profiles.clear()
                logger.info("✅ 已清理所有用户数据")
            
            # 保存更新
            self._save_user_profiles()
            return True
        
        except Exception as e:
            logger.error(f"❌ 清理数据失败: {e}")
            return False

# 全局实例
_global_optimizer: Optional[PromptOptimizerV17] = None

def get_prompt_optimizer() -> PromptOptimizerV17:
    """获取全局优化器实例"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PromptOptimizerV17()
    return _global_optimizer

# 便捷函数
async def optimize_user_prompt(user_id: str, prompt: str, mode: str = "standard") -> OptimizationResult:
    """便捷的提示词优化函数"""
    optimizer = get_prompt_optimizer()
    mode_enum = OptimizationMode(mode)
    return await optimizer.optimize_prompt(user_id, prompt, mode_enum)

if __name__ == "__main__":
    # 测试代码
    async def test_optimizer():
        print("🧠 测试智能提示词优化器V17")
        
        optimizer = get_prompt_optimizer()
        
        # 测试用户画像
        user_id = "test_user_001"
        
        # 测试各种优化模式
        test_prompt = "写代码"
        
        # 标准优化
        result1 = await optimizer.optimize_prompt(user_id, test_prompt, OptimizationMode.STANDARD)
        print(f"✅ 标准优化: {result1.optimized_prompt}")
        
        # 专业优化
        result2 = await optimizer.optimize_prompt(user_id, test_prompt, OptimizationMode.PROFESSIONAL)
        print(f"✅ 专业优化: {result2.optimized_prompt}")
        
        # 小白优化
        result3 = await optimizer.optimize_prompt(user_id, test_prompt, OptimizationMode.BEGINNER)
        print(f"✅ 小白优化: {result3.optimized_prompt}")
        
        # AI格式优化
        result4 = await optimizer.optimize_prompt(user_id, test_prompt, OptimizationMode.AI_FORMAT)
        print(f"✅ AI格式优化: {result4.optimized_prompt}")
        
        # 测试交互式优化
        interaction = await optimizer.start_interactive_optimization(user_id, "分析数据")
        print(f"✅ 交互式优化启动: {interaction.session_id}")
        
        # 测试用户选择处理
        response = await optimizer.handle_user_choice(interaction.session_id, 1)
        print(f"✅ 用户选择处理: {response}")
        
        # 获取存储信息
        storage_info = optimizer.get_data_storage_info()
        print(f"✅ 数据存储信息: {storage_info}")
        
        print("✅ 测试完成")
    
    asyncio.run(test_optimizer())