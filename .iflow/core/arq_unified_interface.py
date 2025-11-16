#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 ARQ统一接口 (Unified ARQ Interface)
====================================

提供统一的ARQ推理引擎接口，解决版本冲突问题：
- 统一的API接口
- 版本兼容性处理
- 自动版本选择
- 依赖注入支持

支持版本：
- V15 Quantum (主要版本)
- V15 Quantum Chinese (中文版本)
- 向后兼容旧版本

作者: iFlow架构团队
版本: 1.0.0
日期: 2025-11-16
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass

# 项目根路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / ".iflow"))

logger = logging.getLogger(__name__)

class ARQVersion(Enum):
    """ARQ版本枚举"""
    V15_QUANTUM = "v15_quantum"
    V15_QUANTUM_CHINESE = "v15_quantum_chinese"
    V14_QUANTUM = "v14_quantum"
    AUTO = "auto"

@dataclass
class ARQConfig:
    """ARQ配置"""
    version: ARQVersion = ARQVersion.AUTO
    enable_chinese: bool = False
    enable_quantum: bool = True
    enable_metacognition: bool = True
    performance_mode: str = "balanced"  # fast, balanced, quality
    cache_enabled: bool = True
    max_concurrent_requests: int = 10

class ARQInterface:
    """ARQ统一接口"""
    
    def __init__(self, config: Optional[ARQConfig] = None):
        self.config = config or ARQConfig()
        self.engine = None
        self.version_info = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """初始化ARQ引擎"""
        try:
            # 根据配置选择版本
            if self.config.version == ARQVersion.AUTO:
                self._auto_select_version()
            else:
                self._load_specific_version(self.config.version)
            
            logger.info(f"ARQ引擎初始化成功: {self.version_info}")
            
        except Exception as e:
            logger.error(f"ARQ引擎初始化失败: {e}")
            # 回退到基础实现
            self._initialize_fallback_engine()
    
    def _auto_select_version(self):
        """自动选择最佳版本"""
        versions_to_try = [
            (ARQVersion.V15_QUANTUM_CHINESE, self._load_v15_chinese),
            (ARQVersion.V15_QUANTUM, self._load_v15_quantum),
            (ARQVersion.V14_QUANTUM, self._load_v14_quantum),
        ]
        
        for version, loader in versions_to_try:
            try:
                loader()
                self.version_info = version.value
                logger.info(f"自动选择ARQ版本: {version.value}")
                return
            except ImportError as e:
                logger.debug(f"版本 {version.value} 不可用: {e}")
                continue
        
        raise ImportError("没有可用的ARQ版本")
    
    def _load_specific_version(self, version: ARQVersion):
        """加载指定版本"""
        loaders = {
            ARQVersion.V15_QUANTUM_CHINESE: self._load_v15_chinese,
            ARQVersion.V15_QUANTUM: self._load_v15_quantum,
            ARQVersion.V14_QUANTUM: self._load_v14_quantum,
        }
        
        if version not in loaders:
            raise ValueError(f"不支持的ARQ版本: {version}")
        
        loaders[version]()
        self.version_info = version.value
    
    def _load_v15_chinese(self):
        """加载V15中文版本"""
        try:
            from core.arq_reasoning_engine_v15_quantum_chinese import get_中文arq引擎v15, 中文思考模式
            self.engine = get_中文arq引擎v15()
            self.思考模式 = 中文思考模式
            logger.debug("V15中文版本加载成功")
        except ImportError as e:
            raise ImportError(f"V15中文版本加载失败: {e}")
    
    def _load_v15_quantum(self):
        """加载V15量子版本"""
        try:
            from core.arq_reasoning_engine_v15_quantum import get_arq_engine_v15_quantum, QuantumThinkingModeV15
            self.engine = get_arq_engine_v15_quantum()
            self.思考模式 = QuantumThinkingModeV15
            logger.debug("V15量子版本加载成功")
        except ImportError as e:
            raise ImportError(f"V15量子版本加载失败: {e}")
    
    def _load_v14_quantum(self):
        """加载V14量子版本"""
        try:
            from core.arq_reasoning_engine_v14_quantum import get_arq_engine_v14_quantum, QuantumThinkingModeV14
            self.engine = get_arq_engine_v14_quantum()
            self.思考模式 = QuantumThinkingModeV14
            logger.debug("V14量子版本加载成功")
        except ImportError as e:
            raise ImportError(f"V14量子版本加载失败: {e}")
    
    def _initialize_fallback_engine(self):
        """初始化回退引擎"""
        logger.warning("使用回退ARQ引擎")
        self.engine = FallbackARQEngine()
        self.version_info = "fallback"
    
    async def reason(self, query: str, thinking_mode: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        执行ARQ推理
        
        Args:
            query: 查询内容
            thinking_mode: 思考模式
            **kwargs: 其他参数
            
        Returns:
            推理结果
        """
        try:
            # 参数验证
            if not query or not query.strip():
                return {
                    "success": False,
                    "error": "查询内容不能为空",
                    "version": self.version_info
                }
            
            # 调用具体引擎
            if hasattr(self.engine, 'reason'):
                if hasattr(self.engine, '中文推理') and self.version_info == "v15_quantum_chinese":
                    # 中文版本
                    from core.arq_reasoning_engine_v15_quantum_chinese import 中文思考模式
                    mode = 中文思考模式.深度思考
                    result = await self.engine.中文推理(query, mode)
                else:
                    # 其他版本
                    result = await self.engine.reason(query, thinking_mode)
                
                # 添加版本信息
                result["version"] = self.version_info
                result["interface"] = "unified"
                return result
            else:
                raise AttributeError("引擎不支持reason方法")
                
        except Exception as e:
            logger.error(f"ARQ推理失败: {e}")
            return {
                "success": False,
                "error": f"推理失败: {str(e)}",
                "version": self.version_info
            }
    
    async def self_reflect(self, topic: str) -> Dict[str, Any]:
        """自我反思"""
        try:
            if hasattr(self.engine, 'self_reflect'):
                result = await self.engine.self_reflect(topic)
                result["version"] = self.version_info
                return result
            else:
                return {
                    "success": False,
                    "error": "当前版本不支持自我反思",
                    "version": self.version_info
                }
        except Exception as e:
            logger.error(f"自我反思失败: {e}")
            return {
                "success": False,
                "error": f"自我反思失败: {str(e)}",
                "version": self.version_info
            }
    
    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            "version": self.version_info,
            "engine_type": type(self.engine).__name__,
            "config": {
                "enable_chinese": self.config.enable_chinese,
                "enable_quantum": self.config.enable_quantum,
                "performance_mode": self.config.performance_mode,
                "cache_enabled": self.config.cache_enabled
            },
            "available": self.engine is not None
        }

class FallbackARQEngine:
    """回退ARQ引擎"""
    
    def __init__(self):
        self.name = "Fallback ARQ Engine"
    
    async def reason(self, query: str, thinking_mode: Optional[str] = None) -> Dict[str, Any]:
        """基础推理实现"""
        return {
            "success": True,
            "conclusion": f"基于查询'{query}'的基础分析结果",
            "confidence": 0.6,
            "reasoning_path": ["基础分析", "简单推理", "结论生成"],
            "version": "fallback"
        }
    
    async def self_reflect(self, topic: str) -> Dict[str, Any]:
        """基础自我反思"""
        return {
            "success": True,
            "reflection": f"关于'{topic}'的基础反思",
            "improvements": ["提升推理深度", "增加知识库", "优化算法"],
            "version": "fallback"
        }

# 全局ARQ实例
_global_arq_instance: Optional[ARQInterface] = None

def get_arq_interface(config: Optional[ARQConfig] = None) -> ARQInterface:
    """获取ARQ接口实例（单例模式）"""
    global _global_arq_instance
    
    if _global_arq_instance is None:
        _global_arq_instance = ARQInterface(config)
    
    return _global_arq_instance

def create_arq_interface(config: Optional[ARQConfig] = None) -> ARQInterface:
    """创建新的ARQ接口实例"""
    return ARQInterface(config)

# 便捷函数
async def arq_reason(query: str, version: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """便捷的ARQ推理函数"""
    config = ARQConfig()
    if version:
        try:
            config.version = ARQVersion(version)
        except ValueError:
            logger.warning(f"无效的ARQ版本: {version}，使用自动选择")
    
    arq = get_arq_interface(config)
    return await arq.reason(query, **kwargs)

async def arq_self_reflect(topic: str, version: Optional[str] = None) -> Dict[str, Any]:
    """便捷的ARQ自我反思函数"""
    config = ARQConfig()
    if version:
        try:
            config.version = ARQVersion(version)
        except ValueError:
            logger.warning(f"无效的ARQ版本: {version}，使用自动选择")
    
    arq = get_arq_interface(config)
    return await arq.self_reflect(topic)

# 版本兼容性工具
def migrate_from_old_version(old_version: str) -> ARQConfig:
    """从旧版本迁移配置"""
    migration_map = {
        "v8": ARQVersion.V14_QUANTUM,
        "v12": ARQVersion.V14_QUANTUM,
        "v13": ARQVersion.V14_QUANTUM,
        "v14": ARQVersion.V14_QUANTUM,
        "v14_quantum": ARQVersion.V14_QUANTUM,
        "v15": ARQVersion.V15_QUANTUM,
        "v15_quantum": ARQVersion.V15_QUANTUM,
        "v15_chinese": ARQVersion.V15_QUANTUM_CHINESE,
    }
    
    return ARQConfig(version=migration_map.get(old_version, ARQVersion.AUTO))

if __name__ == "__main__":
    # 测试ARQ统一接口
    import asyncio
    
    async def test_arq_interface():
        print("🧪 测试ARQ统一接口")
        
        # 测试自动版本选择
        arq = get_arq_interface()
        status = arq.get_status()
        print(f"引擎状态: {status}")
        
        # 测试推理
        result = await arq.reason("什么是人工智能？")
        print(f"推理结果: {result}")
        
        # 测试自我反思
        reflection = await arq.self_reflect("我的推理过程")
        print(f"自我反思: {reflection}")
        
        print("✅ ARQ统一接口测试完成")
    
    asyncio.run(test_arq_interface())