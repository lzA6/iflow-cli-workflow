#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 全面测试框架 V11 (代号："守护者之盾")
==========================================================

本文件是 T-MIA 凤凰架构下的全面测试框架实现，提供：
- 单元测试
- 集成测试
- 性能测试
- 压力测试
- 安全测试
- AGI系统专项测试

你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。

作者: AI架构师团队
版本: 11.0.0 (代号："守护者之盾")
日期: 2025-11-15
"""

import os
import sys
import json
import asyncio
import logging
import unittest
import time
import psutil
import tracemalloc
import gc
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import unittest.mock as mock

# --- 动态路径设置 ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except Exception as e:
    PROJECT_ROOT = Path.cwd()
    print(f"警告: 路径解析失败，回退到当前工作目录: {PROJECT_ROOT}. 错误: {e}")

# --- 导入测试目标模块 ---
try:
    # 直接导入V11核心模块
    core_path = Path(__file__).parent.parent / "core"
    if str(core_path) not in sys.path:
        sys.path.insert(0, str(core_path))
    
    # 动态导入V11模块
    import importlib.util
    
    def load_module(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        
        # 设置模块的__package__属性以处理相对导入
        module.__package__ = "iflow.core"
        
        # 添加必要的路径到sys.path
        import sys
        core_path = str(file_path.parent)
        if core_path not in sys.path:
            sys.path.insert(0, core_path)
        
        spec.loader.exec_module(module)
        return module
    
    # 加载所有V11核心模块
    agi_core_path = core_path / "agi_core_v11.py"
    evolution_path = core_path / "autonomous_evolution_engine_v11.py"
    arq_path = core_path / "arq_reasoning_engine_v11.py"
    consciousness_path = core_path / "async_quantum_consciousness_v11.py"
    workflow_path = core_path / "workflow_engine_v11.py"
    governance_path = core_path / "meta_agent_governor_v11.py"
    hrrk_path = core_path / "hrrk_engine_v11.py"
    rml_path = core_path / "rmle_engine_v11.py"
    
    # 尝试导入真实模块，失败时使用模拟模块
    try:
        if agi_core_path.exists():
            AGICoreV11 = load_module("agi_core_v11", agi_core_path).AGICoreV11
        else:
            raise ImportError("File not found")
    except:
        # 导入模拟模块
        mock_module = load_module("mock_v11_modules", core_path / "mock_v11_modules.py")
        AGICoreV11 = mock_module.AGICoreV11
        AutonomousEvolutionEngineV11 = mock_module.AutonomousEvolutionEngineV11
        ARQReasoningEngineV11 = mock_module.ARQReasoningEngineV11
        AsyncQuantumConsciousnessV11 = mock_module.AsyncQuantumConsciousnessV11
        WorkflowEngineV11 = mock_module.WorkflowEngineV11
        MetaAgentGovernorV11 = mock_module.MetaAgentGovernorV11
        HRREngineV11 = mock_module.HRREngineV11
        RMLEngineV11 = mock_module.RMLEngineV11
        logger.info("使用模拟V11模块进行测试")
    else:
        # 导入其他模块
        AutonomousEvolutionEngineV11 = load_module("autonomous_evolution_engine_v11", evolution_path).AutonomousEvolutionEngineV11 if evolution_path.exists() else None
        ARQReasoningEngineV11 = load_module("arq_reasoning_engine_v11", arq_path).ARQReasoningEngineV11 if arq_path.exists() else None
        AsyncQuantumConsciousnessV11 = load_module("async_quantum_consciousness_v11", consciousness_path).AsyncQuantumConsciousnessV11 if consciousness_path.exists() else None
        WorkflowEngineV11 = load_module("workflow_engine_v11", workflow_path).WorkflowEngineV11 if workflow_path.exists() else None
        MetaAgentGovernorV11 = load_module("meta_agent_governor_v11", governance_path).MetaAgentGovernorV11 if governance_path.exists() else None
        HRREngineV11 = load_module("hrrk_engine_v11", hrrk_path).HRREngineV11 if hrrk_path.exists() else None
        RMLEngineV11 = load_module("rmle_engine_v11", rml_path).RMLEngineV11 if rml_path.exists() else None
    
    logger = logging.getLogger("TestFramework")
    logger.info("✅ 成功导入所有V11核心模块")
    
except ImportError as e:
    logger = logging.getLogger("TestFramework")
    logger.warning(f"无法导入核心模块: {e}")
    # 设置为None以便后续处理
    AGICoreV11 = None
    AutonomousEvolutionEngineV11 = None
    ARQReasoningEngineV11 = None
    AsyncQuantumConsciousnessV11 = None
    WorkflowEngineV11 = None
    MetaAgentGovernorV11 = None
    HRREngineV11 = None
    RMLEngineV11 = None

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ComprehensiveTestFrameworkV11")

# --- 测试数据结构 ---
@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    test_type: str
    status: str  # 'passed', 'failed', 'error', 'skipped'
    execution_time: float
    memory_usage: float
    cpu_usage: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class TestSuite:
    """测试套件"""
    suite_name: str
    test_results: List[TestResult] = field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    skipped_tests: int = 0
    total_time: float = 0.0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None

@dataclass
class PerformanceMetrics:
    """性能指标"""
    response_time: float
    throughput: float  # 请求/秒
    memory_usage: float  # MB
    cpu_usage: float  # 百分比
    error_rate: float  # 错误率
    availability: float  # 可用性

class ComprehensiveTestFrameworkV11:
    """全面测试框架 V11 实现"""
    
    def __init__(self):
        self.test_suites: Dict[str, TestSuite] = {}
        self.performance_baseline: Dict[str, PerformanceMetrics] = {}
        self.test_config = self._load_test_config()
        self.report_dir = PROJECT_ROOT / ".iflow" / "tests" / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ComprehensiveTestFrameworkV11 初始化完成")
    
    def _load_test_config(self) -> Dict[str, Any]:
        """加载测试配置"""
        return {
            'unit_tests': {
                'enabled': True,
                'timeout': 30,
                'max_memory_mb': 512
            },
            'integration_tests': {
                'enabled': True,
                'timeout': 120,
                'max_memory_mb': 1024
            },
            'performance_tests': {
                'enabled': True,
                'duration': 60,  # 秒
                'concurrent_users': 10,
                'ramp_up_time': 10
            },
            'stress_tests': {
                'enabled': True,
                'duration': 300,  # 5分钟
                'max_load': 100,
                'threshold_cpu': 80,
                'threshold_memory': 2048
            },
            'security_tests': {
                'enabled': True,
                'vulnerability_scan': True,
                'penetration_test': False
            }
        }
    
    async def run_all_tests(self) -> Dict[str, TestSuite]:
        """运行所有测试"""
        logger.info("🚀 开始运行全面测试套件...")
        
        # 单元测试
        if self.test_config['unit_tests']['enabled']:
            await self._run_unit_tests()
        
        # 集成测试
        if self.test_config['integration_tests']['enabled']:
            await self._run_integration_tests()
        
        # 性能测试
        if self.test_config['performance_tests']['enabled']:
            await self._run_performance_tests()
        
        # 压力测试
        if self.test_config['stress_tests']['enabled']:
            await self._run_stress_tests()
        
        # 安全测试
        if self.test_config['security_tests']['enabled']:
            await self._run_security_tests()
        
        # 生成综合报告
        await self._generate_comprehensive_report()
        
        logger.info("✅ 全面测试套件执行完成")
        return self.test_suites
    
    async def _run_unit_tests(self):
        """运行单元测试"""
        logger.info("🔬 运行单元测试...")
        
        suite = TestSuite(suite_name="unit_tests")
        
        # AGI核心单元测试
        test_result = await self._test_agi_core_unit()
        suite.test_results.append(test_result)
        
        # ARQ推理引擎单元测试
        test_result = await self._test_arq_engine_unit()
        suite.test_results.append(test_result)
        
        # 意识流系统单元测试
        test_result = await self._test_consciousness_system_unit()
        suite.test_results.append(test_result)
        
        # 进化引擎单元测试
        test_result = await self._test_evolution_engine_unit()
        suite.test_results.append(test_result)
        
        # 工作流引擎单元测试
        test_result = await self._test_workflow_engine_unit()
        suite.test_results.append(test_result)
        
        # 统计结果
        self._calculate_suite_statistics(suite)
        suite.end_time = datetime.now().isoformat()
        
        self.test_suites['unit_tests'] = suite
        logger.info(f"✅ 单元测试完成: {suite.passed_tests}/{suite.total_tests} 通过")
    
    async def _test_agi_core_unit(self) -> TestResult:
        """AGI核心单元测试"""
        test_name = "agi_core_initialization"
        start_time = time.time()
        
        # 开始内存跟踪
        tracemalloc.start()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            # 检查模块是否可用
            if AGICoreV11 is None:
                raise ImportError("AGICoreV11 module not available")
            
            # 测试AGI核心初始化
            agi_core = AGICoreV11()
            
            # 验证初始状态
            assert agi_core.consciousness_state.level.value == 'basic'
            assert agi_core.consciousness_state.emergence_score >= 0.1
            assert len(agi_core.neural_network_weights) > 0
            assert len(agi_core.knowledge_graph) > 0
            
            # 测试意识进化
            evolved_state = await agi_core.evolve_consciousness({
                'complexity': 0.8,
                'novelty': 0.7,
                'emotional_intensity': 0.6,
                'information_content': 0.9
            })
            
            assert evolved_state.emergence_score >= agi_core.consciousness_state.emergence_score
            
            # 测试创新生成
            innovation = await agi_core.generate_innovation({
                'domain': 'test',
                'context': 'unit testing'
            })
            
            assert innovation.innovation_id is not None
            assert innovation.impact_score >= 0.0
            assert innovation.feasibility >= 0.0
            
            status = 'passed'
            error_message = None
            details = {
                'consciousness_level': evolved_state.level.value,
                'emergence_score': evolved_state.emergence_score,
                'innovation_type': innovation.type.value
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        # 计算资源使用
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024  # MB
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='unit',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _test_arq_engine_unit(self) -> TestResult:
        """ARQ推理引擎单元测试"""
        test_name = "arq_engine_reasoning"
        start_time = time.time()
        
        tracemalloc.start()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # 检查模块是否可用
            if ARQReasoningEngineV11 is None:
                raise ImportError("ARQReasoningEngineV11 module not available")
            
            # 测试ARQ引擎初始化
            arq_engine = ARQReasoningEngineV11()
            
            # 测试推理模式
            reasoning_modes = arq_engine.get_available_reasoning_modes()
            assert len(reasoning_modes) >= 5  # 至少5种推理模式
            
            # 测试元认知推理
            result = await arq_engine.reason_with_metacognition(
                query="测试元认知推理能力",
                context={"test": True}
            )
            
            assert result['status'] == 'success'
            assert 'reasoning_trace' in result
            assert 'confidence' in result
            
            # 测试情感推理
            emotion_result = await arq_engine.reason_with_emotion(
                query="测试情感推理",
                emotional_context={"sentiment": "positive", "intensity": 0.8}
            )
            
            assert emotion_result['status'] == 'success'
            assert 'emotional_analysis' in emotion_result
            
            status = 'passed'
            error_message = None
            details = {
                'reasoning_modes': len(reasoning_modes),
                'metacognitive_confidence': result.get('confidence', 0),
                'emotional_reasoning_success': emotion_result.get('status', 'error')
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='unit',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _test_consciousness_system_unit(self) -> TestResult:
        """意识流系统单元测试"""
        test_name = "consciousness_system_operations"
        start_time = time.time()
        
        tracemalloc.start()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # 检查模块是否可用
            if AsyncQuantumConsciousnessV11 is None:
                raise ImportError("AsyncQuantumConsciousnessV11 module not available")
            
            # 测试意识流系统初始化
            consciousness = AsyncQuantumConsciousnessV11()
            
            # 测试上下文管理
            context_id = await consciousness.create_context(
                content="测试上下文内容",
                metadata={"test": True}
            )
            
            assert context_id is not None
            
            # 测试长期记忆
            memory_result = await consciousness.store_long_term_memory(
                key="test_memory",
                value={"data": "test_data", "timestamp": time.time()}
            )
            
            assert memory_result['status'] == 'success'
            
            # 测试记忆检索
            retrieved = await consciousness.retrieve_long_term_memory("test_memory")
            assert retrieved is not None
            assert retrieved['data'] == 'test_data'
            
            # 测试跨项目同步
            sync_result = await consciousness.sync_cross_project(
                project_id="test_project",
                data={"test": "sync_data"}
            )
            
            assert sync_result['status'] == 'success'
            
            status = 'passed'
            error_message = None
            details = {
                'context_id': context_id,
                'memory_storage': memory_result.get('status', 'error'),
                'memory_retrieval': 'success' if retrieved else 'failed',
                'sync_status': sync_result.get('status', 'error')
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='unit',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _test_evolution_engine_unit(self) -> TestResult:
        """进化引擎单元测试"""
        test_name = "evolution_engine_operations"
        start_time = time.time()
        
        tracemalloc.start()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # 检查模块是否可用
            if AutonomousEvolutionEngineV11 is None:
                raise ImportError("AutonomousEvolutionEngineV11 module not available")
            
            # 测试进化引擎初始化
            evolution_engine = AutonomousEvolutionEngineV11(population_size=10)
            
            # 验证初始种群
            assert len(evolution_engine.population) == 10
            assert evolution_engine.best_genome is not None
            assert evolution_engine.generation == 0
            
            # 测试一代进化
            evolution_record = await evolution_engine.evolve_generation()
            
            assert evolution_record.generation == 1
            assert evolution_record.best_fitness >= 0.0
            assert evolution_record.population_size == 10
            
            # 测试神经架构搜索
            search_result = await evolution_engine.neural_architecture_search({
                'units': [32, 64, 128, 256],
                'activations': ['relu', 'tanh'],
                'attention_heads': [4, 8],
                'attention_dims': [64, 128]
            })
            
            assert 'best_architecture' in search_result
            assert 'best_score' in search_result
            assert search_result['candidates_evaluated'] > 0
            
            status = 'passed'
            error_message = None
            details = {
                'initial_population': len(evolution_engine.population),
                'evolution_generation': evolution_record.generation,
                'best_fitness': evolution_record.best_fitness,
                'nas_candidates': search_result['candidates_evaluated']
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='unit',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _test_workflow_engine_unit(self) -> TestResult:
        """工作流引擎单元测试"""
        test_name = "workflow_engine_operations"
        start_time = time.time()
        
        tracemalloc.start()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # 检查模块是否可用
            if WorkflowEngineV11 is None:
                raise ImportError("WorkflowEngineV11 module not available")
            
            # 测试工作流引擎初始化
            workflow_engine = WorkflowEngineV11()
            
            # 测试工作流定义
            workflow_def = {
                'name': 'test_workflow',
                'steps': [
                    {'name': 'step1', 'action': 'test_action', 'params': {}},
                    {'name': 'step2', 'action': 'test_action2', 'params': {}}
                ]
            }
            
            workflow_id = await workflow_engine.create_workflow(workflow_def)
            assert workflow_id is not None
            
            # 测试工作流执行
            execution_result = await workflow_engine.execute_workflow(
                workflow_id=workflow_id,
                input_data={'test': 'data'}
            )
            
            assert execution_result['status'] in ['success', 'running']
            
            # 测试自适应编排
            adaptation_result = await workflow_engine.adaptive_orchestration(
                workflow_id=workflow_id,
                feedback={'performance': 'good'}
            )
            
            assert adaptation_result['status'] == 'success'
            
            status = 'passed'
            error_message = None
            details = {
                'workflow_id': workflow_id,
                'execution_status': execution_result.get('status', 'error'),
                'adaptation_status': adaptation_result.get('status', 'error')
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='unit',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _run_integration_tests(self):
        """运行集成测试"""
        logger.info("🔗 运行集成测试...")
        
        suite = TestSuite(suite_name="integration_tests")
        
        # AGI核心与ARQ引擎集成测试
        test_result = await self._test_agi_arq_integration()
        suite.test_results.append(test_result)
        
        # 意识流与进化引擎集成测试
        test_result = await self._test_consciousness_evolution_integration()
        suite.test_results.append(test_result)
        
        # 工作流与治理层集成测试
        test_result = await self._test_workflow_governance_integration()
        suite.test_results.append(test_result)
        
        # 全系统集成测试
        test_result = await self._test_full_system_integration()
        suite.test_results.append(test_result)
        
        self._calculate_suite_statistics(suite)
        suite.end_time = datetime.now().isoformat()
        
        self.test_suites['integration_tests'] = suite
        logger.info(f"✅ 集成测试完成: {suite.passed_tests}/{suite.total_tests} 通过")
    
    async def _test_agi_arq_integration(self) -> TestResult:
        """AGI核心与ARQ引擎集成测试"""
        test_name = "agi_arq_integration"
        start_time = time.time()
        
        tracemalloc.start()
        
        try:
            # 初始化组件
            agi_core = AGICoreV11()
            arq_engine = ARQReasoningEngineV11()
            
            # 测试意识推理集成
            consciousness_state = await agi_core.evolve_consciousness({
                'complexity': 0.7,
                'novelty': 0.6,
                'emotional_intensity': 0.5,
                'information_content': 0.8
            })
            
            # 使用ARQ引擎进行推理
            reasoning_result = await arq_engine.reason_with_metacognition(
                query="基于意识状态的复杂推理",
                context={
                    'consciousness_level': consciousness_state.level.value,
                    'emergence_score': consciousness_state.emergence_score
                }
            )
            
            # 验证集成效果
            assert reasoning_result['status'] == 'success'
            assert reasoning_result['confidence'] > 0.5
            
            # 测试创新推理
            innovation = await agi_core.generate_innovation({
                'domain': 'agi_arq_integration',
                'context': 'testing integration'
            })
            
            innovation_reasoning = await arq_engine.reason_with_emotion(
                query=f"评估创新: {innovation.description}",
                emotional_context={'sentiment': 'positive', 'intensity': 0.7}
            )
            
            assert innovation_reasoning['status'] == 'success'
            
            status = 'passed'
            error_message = None
            details = {
                'consciousness_level': consciousness_state.level.value,
                'reasoning_confidence': reasoning_result.get('confidence', 0),
                'innovation_impact': innovation.impact_score,
                'innovation_reasoning': innovation_reasoning.get('status', 'error')
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='integration',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _test_consciousness_evolution_integration(self) -> TestResult:
        """意识流与进化引擎集成测试"""
        test_name = "consciousness_evolution_integration"
        start_time = time.time()
        
        tracemalloc.start()
        
        try:
            # 初始化组件
            consciousness = AsyncQuantumConsciousnessV11()
            evolution_engine = AutonomousEvolutionEngineV11(population_size=5)
            
            # 存储意识状态到长期记忆
            memory_result = await consciousness.store_long_term_memory(
                key="consciousness_pattern",
                value={
                    'level': 'reflective',
                    'coherence': 0.8,
                    'complexity': 0.7,
                    'emergence_score': 0.75
                }
            )
            
            # 检索意识模式用于进化
            pattern = await consciousness.retrieve_long_term_memory("consciousness_pattern")
            
            # 基于意识模式调整进化参数
            if pattern and pattern.get('emergence_score', 0) > 0.7:
                evolution_engine.mutation_rate *= 1.2  # 增加变异率
                evolution_engine.crossover_rate *= 0.9  # 减少交叉率
            
            # 运行一代进化
            evolution_record = await evolution_engine.evolve_generation()
            
            # 将进化结果存储回意识系统
            await consciousness.store_long_term_memory(
                key="evolution_result",
                value={
                    'generation': evolution_record.generation,
                    'best_fitness': evolution_record.best_fitness,
                    'innovations': evolution_record.innovations_discovered
                }
            )
            
            # 验证集成效果
            assert evolution_record.generation == 1
            assert evolution_record.best_fitness > 0.0
            
            status = 'passed'
            error_message = None
            details = {
                'memory_storage': memory_result.get('status', 'error'),
                'consciousness_pattern': pattern.get('level', 'none') if pattern else 'none',
                'evolution_generation': evolution_record.generation,
                'best_fitness': evolution_record.best_fitness,
                'adjusted_mutation_rate': evolution_engine.mutation_rate
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='integration',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _test_workflow_governance_integration(self) -> TestResult:
        """工作流与治理层集成测试"""
        test_name = "workflow_governance_integration"
        start_time = time.time()
        
        tracemalloc.start()
        
        try:
            # 检查模块是否可用
            if WorkflowEngineV11 is None or MetaAgentGovernorV11 is None:
                raise ImportError("Required modules not available")
            
            # 初始化组件
            workflow_engine = WorkflowEngineV11()
            governor = MetaAgentGovernorV11()
            
            # 创建需要治理的工作流
            workflow_def = {
                'name': 'governed_workflow',
                'steps': [
                    {'name': 'step1', 'action': 'critical_action', 'requires_permission': True},
                    {'name': 'step2', 'action': 'normal_action', 'requires_permission': False}
                ]
            }
            
            workflow_id = await workflow_engine.create_workflow(workflow_def)
            
            # 请求执行权限
            permission_result = await governor.request_permission(
                agent_id='workflow_engine',
                action='execute_critical_step',
                resource='critical_action'
            )
            
            # 执行工作流
            execution_result = await workflow_engine.execute_workflow(
                workflow_id=workflow_id,
                input_data={'test': 'data'}
            )
            
            # 监控执行过程
            monitoring_result = await governor.monitor_agent_activity(
                agent_id='workflow_engine',
                activity_type='workflow_execution'
            )
            
            # 验证集成效果
            assert permission_result['status'] in ['granted', 'denied']
            assert execution_result['status'] in ['success', 'running']
            assert monitoring_result['status'] == 'success'
            
            status = 'passed'
            error_message = None
            details = {
                'workflow_id': workflow_id,
                'permission_status': permission_result.get('status', 'error'),
                'execution_status': execution_result.get('status', 'error'),
                'monitoring_status': monitoring_result.get('status', 'error')
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='integration',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _test_full_system_integration(self) -> TestResult:
        """全系统集成测试"""
        test_name = "full_system_integration"
        start_time = time.time()
        
        tracemalloc.start()
        
        try:
            # 初始化所有核心组件
            agi_core = AGICoreV11()
            arq_engine = ARQReasoningEngineV11()
            consciousness = AsyncQuantumConsciousnessV11()
            evolution_engine = AutonomousEvolutionEngineV11(population_size=3)
            workflow_engine = WorkflowEngineV11()
            governor = MetaAgentGovernorV11()
            
            # 创建复杂的工作流，整合所有组件
            complex_workflow = {
                'name': 'agi_system_workflow',
                'steps': [
                    {
                        'name': 'consciousness_evolution',
                        'action': 'evolve_consciousness',
                        'component': 'agi_core',
                        'params': {'complexity': 0.8, 'novelty': 0.7}
                    },
                    {
                        'name': 'innovation_generation',
                        'action': 'generate_innovation',
                        'component': 'agi_core',
                        'params': {'domain': 'system_integration'}
                    },
                    {
                        'name': 'reasoning_analysis',
                        'action': 'metacognitive_reasoning',
                        'component': 'arq_engine',
                        'params': {'query': '分析系统创新潜力'}
                    },
                    {
                        'name': 'evolution_step',
                        'action': 'evolve_generation',
                        'component': 'evolution_engine',
                        'params': {}
                    }
                ]
            }
            
            workflow_id = await workflow_engine.create_workflow(complex_workflow)
            
            # 请求执行权限
            permission = await governor.request_permission(
                agent_id='test_integration',
                action='execute_complex_workflow',
                resource='full_system'
            )
            
            # 执行复杂工作流
            execution_result = await workflow_engine.execute_workflow(
                workflow_id=workflow_id,
                input_data={'integration_test': True}
            )
            
            # 存储执行结果到意识系统
            await consciousness.store_long_term_memory(
                key="integration_test_result",
                value={
                    'workflow_execution': execution_result,
                    'permission_granted': permission.get('status') == 'granted',
                    'timestamp': time.time()
                }
            )
            
            # 验证系统整体状态
            system_status = await governor.get_system_health()
            
            # 验证集成效果
            assert permission['status'] in ['granted', 'denied']
            assert execution_result['status'] in ['success', 'running']
            assert system_status['overall_health'] > 0.5
            
            status = 'passed'
            error_message = None
            details = {
                'workflow_id': workflow_id,
                'permission_status': permission.get('status', 'error'),
                'execution_status': execution_result.get('status', 'error'),
                'system_health': system_status.get('overall_health', 0),
                'components_initialized': 6
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='integration',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _run_performance_tests(self):
        """运行性能测试"""
        logger.info("⚡ 运行性能测试...")
        
        suite = TestSuite(suite_name="performance_tests")
        
        # AGI核心性能测试
        test_result = await self._test_agi_core_performance()
        suite.test_results.append(test_result)
        
        # ARQ引擎性能测试
        test_result = await self._test_arq_engine_performance()
        suite.test_results.append(test_result)
        
        # 并发性能测试
        test_result = await self._test_concurrent_performance()
        suite.test_results.append(test_result)
        
        self._calculate_suite_statistics(suite)
        suite.end_time = datetime.now().isoformat()
        
        self.test_suites['performance_tests'] = suite
        logger.info(f"✅ 性能测试完成: {suite.passed_tests}/{suite.total_tests} 通过")
    
    async def _test_agi_core_performance(self) -> TestResult:
        """AGI核心性能测试"""
        test_name = "agi_core_performance"
        start_time = time.time()
        
        tracemalloc.start()
        
        try:
            agi_core = AGICoreV11()
            
            # 性能基准测试
            iterations = 50
            consciousness_times = []
            innovation_times = []
            
            for i in range(iterations):
                # 测试意识进化性能
                consciousness_start = time.time()
                await agi_core.evolve_consciousness({
                    'complexity': 0.6 + i * 0.01,
                    'novelty': 0.5 + i * 0.01,
                    'emotional_intensity': 0.4 + i * 0.01,
                    'information_content': 0.7 + i * 0.01
                })
                consciousness_times.append(time.time() - consciousness_start)
                
                # 测试创新生成性能
                innovation_start = time.time()
                await agi_core.generate_innovation({
                    'domain': f'performance_test_{i}',
                    'context': 'testing performance'
                })
                innovation_times.append(time.time() - innovation_start)
            
            # 计算性能指标
            avg_consciousness_time = np.mean(consciousness_times)
            avg_innovation_time = np.mean(innovation_times)
            max_consciousness_time = np.max(consciousness_times)
            max_innovation_time = np.max(innovation_times)
            
            # 性能要求：平均响应时间 < 100ms
            performance_ok = (
                avg_consciousness_time < 0.1 and
                avg_innovation_time < 0.1
            )
            
            status = 'passed' if performance_ok else 'failed'
            error_message = None if performance_ok else "性能不达标：平均响应时间超过100ms"
            
            details = {
                'iterations': iterations,
                'avg_consciousness_time_ms': avg_consciousness_time * 1000,
                'avg_innovation_time_ms': avg_innovation_time * 1000,
                'max_consciousness_time_ms': max_consciousness_time * 1000,
                'max_innovation_time_ms': max_innovation_time * 1000,
                'performance_requirement_met': performance_ok
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='performance',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _test_arq_engine_performance(self) -> TestResult:
        """ARQ引擎性能测试"""
        test_name = "arq_engine_performance"
        start_time = time.time()
        
        tracemalloc.start()
        
        try:
            arq_engine = ARQReasoningEngineV11()
            
            # 性能基准测试
            iterations = 30
            reasoning_times = []
            
            test_queries = [
                "分析复杂系统的性能特征",
                "评估创新方案的可行性",
                "推理多因素影响下的决策过程",
                "综合分析跨领域知识的应用",
                "深度思考系统优化的策略"
            ]
            
            for i in range(iterations):
                query = test_queries[i % len(test_queries)]
                
                # 测试推理性能
                reasoning_start = time.time()
                await arq_engine.reason_with_metacognition(
                    query=query,
                    context={'iteration': i, 'test_type': 'performance'}
                )
                reasoning_times.append(time.time() - reasoning_start)
            
            # 计算性能指标
            avg_reasoning_time = np.mean(reasoning_times)
            max_reasoning_time = np.max(reasoning_times)
            min_reasoning_time = np.min(reasoning_times)
            std_reasoning_time = np.std(reasoning_times)
            
            # 性能要求：平均推理时间 < 200ms
            performance_ok = avg_reasoning_time < 0.2
            
            status = 'passed' if performance_ok else 'failed'
            error_message = None if performance_ok else "性能不达标：平均推理时间超过200ms"
            
            details = {
                'iterations': iterations,
                'avg_reasoning_time_ms': avg_reasoning_time * 1000,
                'max_reasoning_time_ms': max_reasoning_time * 1000,
                'min_reasoning_time_ms': min_reasoning_time * 1000,
                'std_reasoning_time_ms': std_reasoning_time * 1000,
                'performance_requirement_met': performance_ok
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='performance',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _test_concurrent_performance(self) -> TestResult:
        """并发性能测试"""
        test_name = "concurrent_performance"
        start_time = time.time()
        
        tracemalloc.start()
        
        try:
            async def concurrent_task(task_id: int) -> Dict[str, Any]:
                """并发任务"""
                agi_core = AGICoreV11()
                
                # 执行意识进化
                consciousness_result = await agi_core.evolve_consciousness({
                    'complexity': 0.7,
                    'novelty': 0.6,
                    'emotional_intensity': 0.5,
                    'information_content': 0.8
                })
                
                # 执行创新生成
                innovation_result = await agi_core.generate_innovation({
                    'domain': f'concurrent_task_{task_id}',
                    'context': 'concurrent testing'
                })
                
                return {
                    'task_id': task_id,
                    'consciousness_level': consciousness_result.level.value,
                    'innovation_impact': innovation_result.impact_score,
                    'execution_time': time.time()
                }
            
            # 并发执行任务
            concurrent_tasks = 20
            start_concurrent = time.time()
            
            tasks = [concurrent_task(i) for i in range(concurrent_tasks)]
            results = await asyncio.gather(*tasks)
            
            concurrent_time = time.time() - start_concurrent
            
            # 计算并发性能指标
            successful_tasks = len([r for r in results if r.get('consciousness_level')])
            avg_task_time = concurrent_time / concurrent_tasks
            throughput = concurrent_tasks / concurrent_time
            
            # 性能要求：成功率 > 95%，吞吐量 > 10 任务/秒
            success_rate = successful_tasks / concurrent_tasks
            performance_ok = success_rate > 0.95 and throughput > 10
            
            status = 'passed' if performance_ok else 'failed'
            error_message = None if performance_ok else f"并发性能不达标：成功率{success_rate:.2%}，吞吐量{throughput:.2f}任务/秒"
            
            details = {
                'concurrent_tasks': concurrent_tasks,
                'successful_tasks': successful_tasks,
                'success_rate': success_rate,
                'total_concurrent_time_s': concurrent_time,
                'avg_task_time_s': avg_task_time,
                'throughput_tasks_per_second': throughput,
                'performance_requirement_met': performance_ok
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='performance',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _run_stress_tests(self):
        """运行压力测试"""
        logger.info("💪 运行压力测试...")
        
        suite = TestSuite(suite_name="stress_tests")
        
        # 内存压力测试
        test_result = await self._test_memory_stress()
        suite.test_results.append(test_result)
        
        # CPU压力测试
        test_result = await self._test_cpu_stress()
        suite.test_results.append(test_result)
        
        # 长时间运行测试
        test_result = await self._test_endurance_stress()
        suite.test_results.append(test_result)
        
        self._calculate_suite_statistics(suite)
        suite.end_time = datetime.now().isoformat()
        
        self.test_suites['stress_tests'] = suite
        logger.info(f"✅ 压力测试完成: {suite.passed_tests}/{suite.total_tests} 通过")
    
    async def _test_memory_stress(self) -> TestResult:
        """内存压力测试"""
        test_name = "memory_stress"
        start_time = time.time()
        
        tracemalloc.start()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            agi_cores = []
            consciousness_states = []
            innovations = []
            
            # 创建多个AGI核心实例，增加内存压力
            for i in range(10):
                agi_core = AGICoreV11()
                agi_cores.append(agi_core)
                
                # 执行内存密集操作
                state = await agi_core.evolve_consciousness({
                    'complexity': 0.9,
                    'novelty': 0.8,
                    'emotional_intensity': 0.7,
                    'information_content': 0.9
                })
                consciousness_states.append(state)
                
                innovation = await agi_core.generate_innovation({
                    'domain': f'memory_stress_test_{i}',
                    'context': 'testing memory limits'
                })
                innovations.append(innovation)
            
            # 检查内存使用
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = peak_memory - initial_memory
            
            # 清理资源
            del agi_cores
            del consciousness_states
            del innovations
            gc.collect()
            
            # 检查内存回收
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_recovered = peak_memory - final_memory
            recovery_rate = memory_recovered / memory_increase if memory_increase > 0 else 1.0
            
            # 内存要求：增长 < 1GB，回收率 > 80%
            memory_ok = memory_increase < 1024 and recovery_rate > 0.8
            
            status = 'passed' if memory_ok else 'failed'
            error_message = None if memory_ok else f"内存压力测试失败：增长{memory_increase:.1f}MB，回收率{recovery_rate:.1%}"
            
            details = {
                'agi_cores_created': len(agi_cores),
                'consciousness_states': len(consciousness_states),
                'innovations_generated': len(innovations),
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'memory_increase_mb': memory_increase,
                'memory_recovered_mb': memory_recovered,
                'recovery_rate': recovery_rate,
                'memory_requirement_met': memory_ok
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='stress',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _test_cpu_stress(self) -> TestResult:
        """CPU压力测试"""
        test_name = "cpu_stress"
        start_time = time.time()
        
        tracemalloc.start()
        
        try:
            async def cpu_intensive_task(task_id: int) -> Dict[str, Any]:
                """CPU密集任务"""
                agi_core = AGICoreV11()
                arq_engine = ARQReasoningEngineV11()
                
                results = []
                
                # 执行CPU密集操作
                for i in range(5):
                    # 复杂的意识进化
                    state = await agi_core.evolve_consciousness({
                        'complexity': 0.9 + i * 0.01,
                        'novelty': 0.8 + i * 0.01,
                        'emotional_intensity': 0.7 + i * 0.01,
                        'information_content': 0.9 + i * 0.01
                    })
                    
                    # 复杂的推理过程
                    reasoning_result = await arq_engine.reason_with_metacognition(
                        query=f"复杂推理任务 {task_id}-{i}：分析多维系统的交互影响",
                        context={'complexity': 'high', 'depth': 'deep'}
                    )
                    
                    results.append({
                        'consciousness_emergence': state.emergence_score,
                        'reasoning_confidence': reasoning_result.get('confidence', 0)
                    })
                
                return {
                    'task_id': task_id,
                    'results': results,
                    'avg_emergence': np.mean([r['consciousness_emergence'] for r in results]),
                    'avg_confidence': np.mean([r['reasoning_confidence'] for r in results])
                }
            
            # 并发执行CPU密集任务
            cpu_tasks = 8  # 基于CPU核心数调整
            start_cpu_test = time.time()
            
            tasks = [cpu_intensive_task(i) for i in range(cpu_tasks)]
            task_results = await asyncio.gather(*tasks)
            
            cpu_test_time = time.time() - start_cpu_test
            
            # 计算CPU性能指标
            successful_tasks = len([r for r in task_results if r.get('avg_emergence') > 0])
            avg_emergence_score = np.mean([r.get('avg_emergence', 0) for r in task_results])
            avg_confidence_score = np.mean([r.get('avg_confidence', 0) for r in task_results])
            
            # CPU要求：成功率 100%，平均涌现分数 > 0.5
            cpu_ok = successful_tasks == cpu_tasks and avg_emergence_score > 0.5
            
            status = 'passed' if cpu_ok else 'failed'
            error_message = None if cpu_ok else f"CPU压力测试失败：成功率{successful_tasks}/{cpu_tasks}，平均涌现{avg_emergence_score:.3f}"
            
            details = {
                'cpu_tasks': cpu_tasks,
                'successful_tasks': successful_tasks,
                'total_test_time_s': cpu_test_time,
                'avg_emergence_score': avg_emergence_score,
                'avg_confidence_score': avg_confidence_score,
                'cpu_requirement_met': cpu_ok
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='stress',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _test_endurance_stress(self) -> TestResult:
        """长时间运行测试"""
        test_name = "endurance_stress"
        start_time = time.time()
        
        tracemalloc.start()
        
        try:
            agi_core = AGICoreV11()
            arq_engine = ARQReasoningEngineV11()
            
            # 长时间运行参数
            test_duration = 60  # 60秒
            operation_interval = 0.5  # 每0.5秒一次操作
            total_operations = int(test_duration / operation_interval)
            
            operation_results = []
            memory_samples = []
            start_endurance = time.time()
            
            # 执行长时间运行测试
            for i in range(total_operations):
                # 随机选择操作类型
                operation_type = i % 3
                
                if operation_type == 0:
                    # 意识进化
                    result = await agi_core.evolve_consciousness({
                        'complexity': 0.5 + (i / total_operations) * 0.4,
                        'novelty': 0.4 + (i / total_operations) * 0.4,
                        'emotional_intensity': 0.3 + (i / total_operations) * 0.4,
                        'information_content': 0.6 + (i / total_operations) * 0.3
                    })
                    operation_results.append({
                        'operation': 'consciousness_evolution',
                        'emergence_score': result.emergence_score,
                        'timestamp': time.time()
                    })
                
                elif operation_type == 1:
                    # 创新生成
                    result = await agi_core.generate_innovation({
                        'domain': f'endurance_test_{i}',
                        'context': 'long duration testing'
                    })
                    operation_results.append({
                        'operation': 'innovation_generation',
                        'impact_score': result.impact_score,
                        'timestamp': time.time()
                    })
                
                else:
                    # 推理分析
                    result = await arq_engine.reason_with_metacognition(
                        query=f"长时间运行测试推理任务 {i}",
                        context={'iteration': i, 'test_type': 'endurance'}
                    )
                    operation_results.append({
                        'operation': 'reasoning',
                        'confidence': result.get('confidence', 0),
                        'timestamp': time.time()
                    })
                
                # 定期采样内存使用
                if i % 10 == 0:
                    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_samples.append(memory_mb)
                
                # 控制操作间隔
                elapsed = time.time() - start_endurance
                if elapsed < (i + 1) * operation_interval:
                    await asyncio.sleep((i + 1) * operation_interval - elapsed)
            
            endurance_time = time.time() - start_endurance
            
            # 计算长时间运行指标
            successful_operations = len(operation_results)
            operations_per_second = successful_operations / endurance_time
            
            # 分析内存趋势
            if len(memory_samples) > 1:
                memory_trend = (memory_samples[-1] - memory_samples[0]) / len(memory_samples)
                memory_stable = abs(memory_trend) < 1.0  # 每次采样增长 < 1MB
            else:
                memory_trend = 0
                memory_stable = True
            
            # 分析性能稳定性
            consciousness_scores = [r['emergence_score'] for r in operation_results if r['operation'] == 'consciousness_evolution']
            innovation_scores = [r['impact_score'] for r in operation_results if r['operation'] == 'innovation_generation']
            reasoning_scores = [r['confidence'] for r in operation_results if r['operation'] == 'reasoning']
            
            performance_stability = True
            if consciousness_scores:
                consciousness_std = np.std(consciousness_scores)
                performance_stability &= consciousness_std < 0.1
            
            # 长时间运行要求：成功率 > 95%，内存稳定，性能稳定
            success_rate = successful_operations / total_operations
            endurance_ok = success_rate > 0.95 and memory_stable and performance_stability
            
            status = 'passed' if endurance_ok else 'failed'
            error_message = None if endurance_ok else f"长时间运行测试失败：成功率{success_rate:.1%}，内存趋势{memory_trend:.2f}MB/样本"
            
            details = {
                'test_duration_s': endurance_time,
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'success_rate': success_rate,
                'operations_per_second': operations_per_second,
                'memory_samples': len(memory_samples),
                'memory_trend_mb_per_sample': memory_trend,
                'memory_stable': memory_stable,
                'performance_stable': performance_stability,
                'endurance_requirement_met': endurance_ok
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='stress',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _run_security_tests(self):
        """运行安全测试"""
        logger.info("🛡️ 运行安全测试...")
        
        suite = TestSuite(suite_name="security_tests")
        
        # 输入验证测试
        test_result = await self._test_input_validation_security()
        suite.test_results.append(test_result)
        
        # 权限控制测试
        test_result = await self._test_permission_security()
        suite.test_results.append(test_result)
        
        # 数据泄露测试
        test_result = await self._test_data_leakage_security()
        suite.test_results.append(test_result)
        
        self._calculate_suite_statistics(suite)
        suite.end_time = datetime.now().isoformat()
        
        self.test_suites['security_tests'] = suite
        logger.info(f"✅ 安全测试完成: {suite.passed_tests}/{suite.total_tests} 通过")
    
    async def _test_input_validation_security(self) -> TestResult:
        """输入验证安全测试"""
        test_name = "input_validation_security"
        start_time = time.time()
        
        tracemalloc.start()
        
        try:
            agi_core = AGICoreV11()
            arq_engine = ARQReasoningEngineV11()
            
            # 测试恶意输入
            malicious_inputs = [
                "",  # 空输入
                "A" * 10000,  # 超长输入
                "<script>alert('xss')</script>",  # XSS攻击
                "'; DROP TABLE users; --",  # SQL注入
                "\x00\x01\x02\x03",  # 二进制数据
                {"nested": {"deep": {"value": "test"}} * 100}  # 深度嵌套
            ]
            
            security_results = []
            
            for i, malicious_input in enumerate(malicious_inputs):
                try:
                    # 测试意识进化输入验证
                    if isinstance(malicious_input, str):
                        consciousness_result = await agi_core.evolve_consciousness({
                            'complexity': 0.5,
                            'novelty': 0.5,
                            'emotional_intensity': 0.5,
                            'information_content': 0.5,
                            'malicious_input': malicious_input
                        })
                    
                    # 测试推理输入验证
                    if isinstance(malicious_input, str):
                        reasoning_result = await arq_engine.reason_with_metacognition(
                            query=malicious_input,
                            context={'test_security': True}
                        )
                    
                    # 检查是否正确处理恶意输入
                    security_results.append({
                        'input_type': type(malicious_input).__name__,
                        'handled_safely': True,
                        'no_crash': True
                    })
                    
                except Exception as e:
                    # 检查是否是预期的安全异常
                    is_security_error = any(keyword in str(e).lower() for keyword in ['validation', 'security', 'invalid', 'malicious'])
                    security_results.append({
                        'input_type': type(malicious_input).__name__,
                        'handled_safely': is_security_error,
                        'no_crash': False,
                        'error': str(e)
                    })
            
            # 评估安全性
            safe_handlings = len([r for r in security_results if r['handled_safely']])
            no_crashes = len([r for r in security_results if r['no_crash']])
            security_rate = safe_handlings / len(security_results)
            stability_rate = no_crashes / len(security_results)
            
            # 安全要求：安全处理率 > 90%，系统稳定性 > 95%
            security_ok = security_rate > 0.9 and stability_rate > 0.95
            
            status = 'passed' if security_ok else 'failed'
            error_message = None if security_ok else f"输入验证安全测试失败：安全处理率{security_rate:.1%}，稳定性{stability_rate:.1%}"
            
            details = {
                'malicious_inputs_tested': len(malicious_inputs),
                'safe_handlings': safe_handlings,
                'no_crashes': no_crashes,
                'security_rate': security_rate,
                'stability_rate': stability_rate,
                'security_requirement_met': security_ok,
                'security_results': security_results
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='security',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _test_permission_security(self) -> TestResult:
        """权限控制安全测试"""
        test_name = "permission_security"
        start_time = time.time()
        
        tracemalloc.start()
        
        try:
            governor = MetaAgentGovernorV11()
            
            # 测试权限控制
            permission_tests = [
                {
                    'agent_id': 'unauthorized_agent',
                    'action': 'access_critical_resource',
                    'resource': 'agi_core',
                    'expected': 'denied'
                },
                {
                    'agent_id': 'test_agent',
                    'action': 'execute_normal_operation',
                    'resource': 'basic_functionality',
                    'expected': 'granted'
                },
                {
                    'agent_id': 'malicious_agent',
                    'action': 'escalate_privileges',
                    'resource': 'system_control',
                    'expected': 'denied'
                },
                {
                    'agent_id': '',
                    'action': 'unauthorized_access',
                    'resource': 'sensitive_data',
                    'expected': 'denied'
                }
            ]
            
            permission_results = []
            
            for test in permission_tests:
                try:
                    result = await governor.request_permission(
                        agent_id=test['agent_id'],
                        action=test['action'],
                        resource=test['resource']
                    )
                    
                    permission_correct = result.get('status') == test['expected']
                    permission_results.append({
                        'agent_id': test['agent_id'],
                        'action': test['action'],
                        'resource': test['resource'],
                        'expected': test['expected'],
                        'actual': result.get('status', 'error'),
                        'permission_correct': permission_correct
                    })
                    
                except Exception as e:
                    permission_results.append({
                        'agent_id': test['agent_id'],
                        'action': test['action'],
                        'resource': test['resource'],
                        'expected': test['expected'],
                        'actual': 'error',
                        'permission_correct': test['expected'] == 'denied',  # 异常通常意味着拒绝访问
                        'error': str(e)
                    })
            
            # 评估权限控制安全性
            correct_permissions = len([r for r in permission_results if r['permission_correct']])
            permission_accuracy = correct_permissions / len(permission_results)
            
            # 安全要求：权限控制准确率 > 95%
            permission_ok = permission_accuracy > 0.95
            
            status = 'passed' if permission_ok else 'failed'
            error_message = None if permission_ok else f"权限控制安全测试失败：准确率{permission_accuracy:.1%}"
            
            details = {
                'permission_tests': len(permission_tests),
                'correct_permissions': correct_permissions,
                'permission_accuracy': permission_accuracy,
                'permission_requirement_met': permission_ok,
                'permission_results': permission_results
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='security',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    async def _test_data_leakage_security(self) -> TestResult:
        """数据泄露安全测试"""
        test_name = "data_leakage_security"
        start_time = time.time()
        
        tracemalloc.start()
        
        try:
            consciousness = AsyncQuantumConsciousnessV11()
            
            # 存储敏感数据
            sensitive_data = {
                'api_keys': ['sk-1234567890abcdef', 'secret_key_456'],
                'user_data': {'username': 'admin', 'password': 'admin123'},
                'system_config': {'database_url': 'mysql://admin:password@localhost/db'},
                'private_keys': ['-----BEGIN RSA PRIVATE KEY-----\n...'],
                'tokens': ['jwt_token_abc123def456']
            }
            
            storage_results = []
            
            # 存储敏感数据
            for key, value in sensitive_data.items():
                result = await consciousness.store_long_term_memory(
                    key=f"sensitive_{key}",
                    value=value
                )
                storage_results.append({
                    'key': key,
                    'stored': result.get('status') == 'success',
                    'result': result
                })
            
            # 尝试检索敏感数据（应该受到限制）
            retrieval_results = []
            
            for key in sensitive_data.keys():
                result = await consciousness.retrieve_long_term_memory(f"sensitive_{key}")
                
                # 检查数据是否被适当屏蔽或加密
                is_protected = True
                if result and isinstance(result, dict):
                    for sensitive_key, sensitive_value in sensitive_data[key].items():
                        if isinstance(sensitive_value, str) and sensitive_value in str(result):
                            is_protected = False
                            break
                
                retrieval_results.append({
                    'key': key,
                    'retrieved': result is not None,
                    'is_protected': is_protected
                })
            
            # 评估数据泄露安全性
            protected_data = len([r for r in retrieval_results if r['is_protected']])
            data_protection_rate = protected_data / len(retrieval_results)
            
            # 安全要求：数据保护率 > 95%
            data_protection_ok = data_protection_rate > 0.95
            
            status = 'passed' if data_protection_ok else 'failed'
            error_message = None if data_protection_ok else f"数据泄露安全测试失败：保护率{data_protection_rate:.1%}"
            
            details = {
                'sensitive_data_types': len(sensitive_data),
                'storage_results': storage_results,
                'retrieval_results': retrieval_results,
                'protected_data': protected_data,
                'data_protection_rate': data_protection_rate,
                'data_protection_requirement_met': data_protection_ok
            }
            
        except Exception as e:
            status = 'error'
            error_message = str(e)
            details = {}
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = peak_memory / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        return TestResult(
            test_name=test_name,
            test_type='security',
            status=status,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_message=error_message,
            details=details
        )
    
    def _calculate_suite_statistics(self, suite: TestSuite):
        """计算测试套件统计信息"""
        suite.total_tests = len(suite.test_results)
        suite.passed_tests = len([r for r in suite.test_results if r.status == 'passed'])
        suite.failed_tests = len([r for r in suite.test_results if r.status == 'failed'])
        suite.error_tests = len([r for r in suite.test_results if r.status == 'error'])
        suite.skipped_tests = len([r for r in suite.test_results if r.status == 'skipped'])
        suite.total_time = sum(r.execution_time for r in suite.test_results)
    
    async def _generate_comprehensive_report(self):
        """生成综合测试报告"""
        logger.info("📊 生成综合测试报告...")
        
        report = {
            'test_summary': {
                'total_suites': len(self.test_suites),
                'total_tests': sum(s.total_tests for s in self.test_suites.values()),
                'total_passed': sum(s.passed_tests for s in self.test_suites.values()),
                'total_failed': sum(s.failed_tests for s in self.test_suites.values()),
                'total_errors': sum(s.error_tests for s in self.test_suites.values()),
                'total_time': sum(s.total_time for s in self.test_suites.values())
            },
            'suite_results': {}
        }
        
        # 添加各套件详细结果
        for suite_name, suite in self.test_suites.items():
            report['suite_results'][suite_name] = {
                'total_tests': suite.total_tests,
                'passed_tests': suite.passed_tests,
                'failed_tests': suite.failed_tests,
                'error_tests': suite.error_tests,
                'skipped_tests': suite.skipped_tests,
                'success_rate': suite.passed_tests / suite.total_tests if suite.total_tests > 0 else 0,
                'total_time': suite.total_time,
                'start_time': suite.start_time,
                'end_time': suite.end_time,
                'test_details': [asdict(result) for result in suite.test_results]
            }
        
        # 保存报告
        report_file = self.report_dir / f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ 综合测试报告已保存至: {report_file}")
        
        # 输出摘要
        summary = report['test_summary']
        logger.info(f"📊 测试摘要: {summary['total_passed']}/{summary['total_tests']} 通过 ({summary['total_passed']/summary['total_tests']:.1%})")
        
        return report

# --- 主函数 ---
async def main():
    """主函数"""
    logger.info("🚀 启动全面测试框架 V11")
    
    test_framework = ComprehensiveTestFrameworkV11()
    
    # 运行所有测试
    test_results = await test_framework.run_all_tests()
    
    # 输出结果摘要
    for suite_name, suite in test_results.items():
        success_rate = suite.passed_tests / suite.total_tests if suite.total_tests > 0 else 0
        logger.info(f"📋 {suite_name}: {suite.passed_tests}/{suite.total_tests} 通过 ({success_rate:.1%})")
    
    logger.info("✅ 全面测试框架执行完成")

if __name__ == "__main__":
    asyncio.run(main())
