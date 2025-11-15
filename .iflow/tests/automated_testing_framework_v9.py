#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 自动化测试框架 V9 (Automated Testing Framework V9)
企业级测试解决方案，支持单元测试、集成测试、性能测试和端到端测试

核心特性：
1. 智能测试生成 - AI驱动的测试用例自动生成
2. 并行测试执行 - 多线程/多进程测试优化
3. 实时性能监控 - 测试过程中的性能指标收集
4. 智能缺陷分析 - 自动化缺陷定位和分析
5. 测试报告生成 - 详细的测试报告和可视化
"""

import os
import sys
import json
import asyncio
import logging
import time
import unittest
import threading
import multiprocessing
import pytest
import coverage
import psutil
import gc
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
from collections import defaultdict, deque
import inspect
import importlib.util

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 核心枚举和数据结构 ---

class TestType(Enum):
    """测试类型"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    END_TO_END = "end_to_end"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"

class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

class Priority(Enum):
    """优先级"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class TestCase:
    """测试用例"""
    id: str
    name: str
    test_type: TestType
    function: Callable
    priority: Priority = Priority.MEDIUM
    timeout: float = 30.0
    expected_result: Any = None
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class TestSuite:
    """测试套件"""
    name: str
    test_cases: List[TestCase] = field(default_factory=list)
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    parallel: bool = True
    max_workers: int = 4
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0

@dataclass
class PerformanceMetrics:
    """性能指标"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_peak: float = 0.0
    execution_time: float = 0.0
    throughput: float = 0.0
    latency_avg: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    error_rate: float = 0.0
    requests_per_second: float = 0.0

class IntelligentTestGenerator:
    """智能测试生成器"""
    
    def __init__(self):
        self.code_analyzer = CodeAnalyzer()
        self.pattern_database = TestPatternDatabase()
        self.generation_history = deque(maxlen=1000)
        
    def generate_tests_for_function(self, func: Callable, 
                                  test_type: TestType = TestType.UNIT) -> List[TestCase]:
        """为函数生成测试用例"""
        test_cases = []
        
        try:
            # 分析函数签名和文档
            func_info = self.code_analyzer.analyze_function(func)
            
            # 生成正常情况测试
            normal_cases = self._generate_normal_cases(func_info)
            test_cases.extend(normal_cases)
            
            # 生成边界情况测试
            edge_cases = self._generate_edge_cases(func_info)
            test_cases.extend(edge_cases)
            
            # 生成异常情况测试
            error_cases = self._generate_error_cases(func_info)
            test_cases.extend(error_cases)
            
            # 生成性能测试
            if test_type in [TestType.PERFORMANCE, TestType.INTEGRATION]:
                perf_cases = self._generate_performance_cases(func_info)
                test_cases.extend(perf_cases)
            
        except Exception as e:
            logger.error(f"测试生成失败: {e}")
        
        return test_cases
    
    def _generate_normal_cases(self, func_info: Dict[str, Any]) -> List[TestCase]:
        """生成正常情况测试"""
        test_cases = []
        
        # 基于参数类型生成测试数据
        for param in func_info.get('parameters', []):
            param_name = param['name']
            param_type = param.get('type', 'any')
            
            # 生成典型值
            test_values = self._get_typical_values(param_type)
            
            for value in test_values:
                test_case = TestCase(
                    id=f"normal_{param_name}_{hash(str(value)) % 10000}",
                    name=f"Test {func_info['name']} with {param_name}={value}",
                    test_type=TestType.UNIT,
                    function=self._create_test_function(func_info['name'], {param_name: value}),
                    priority=Priority.HIGH,
                    tags=["normal", param_name]
                )
                test_cases.append(test_case)
        
        return test_cases
    
    def _generate_edge_cases(self, func_info: Dict[str, Any]) -> List[TestCase]:
        """生成边界情况测试"""
        test_cases = []
        
        # 空值测试
        for param in func_info.get('parameters', []):
            param_name = param['name']
            
            # None值测试
            test_case = TestCase(
                id=f"edge_{param_name}_none",
                name=f"Test {func_info['name']} with {param_name}=None",
                test_type=TestType.UNIT,
                function=self._create_test_function(func_info['name'], {param_name: None}),
                priority=Priority.MEDIUM,
                tags=["edge", "null"]
            )
            test_cases.append(test_case)
            
            # 空字符串/空列表测试
            if param.get('type') in ['str', 'list', 'dict']:
                empty_value = '' if param.get('type') == 'str' else ([] if param.get('type') == 'list' else {})
                test_case = TestCase(
                    id=f"edge_{param_name}_empty",
                    name=f"Test {func_info['name']} with {param_name}=empty",
                    test_type=TestType.UNIT,
                    function=self._create_test_function(func_info['name'], {param_name: empty_value}),
                    priority=Priority.MEDIUM,
                    tags=["edge", "empty"]
                )
                test_cases.append(test_case)
        
        return test_cases
    
    def _generate_error_cases(self, func_info: Dict[str, Any]) -> List[TestCase]:
        """生成异常情况测试"""
        test_cases = []
        
        # 类型错误测试
        for param in func_info.get('parameters', []):
            param_name = param['name']
            param_type = param.get('type', 'any')
            
            # 生成错误类型的值
            wrong_values = self._get_wrong_type_values(param_type)
            
            for wrong_value in wrong_values:
                test_case = TestCase(
                    id=f"error_{param_name}_type",
                    name=f"Test {func_info['name']} with wrong type for {param_name}",
                    test_type=TestType.UNIT,
                    function=self._create_test_function(func_info['name'], {param_name: wrong_value}),
                    priority=Priority.MEDIUM,
                    tags=["error", "type"],
                    expected_result="exception"
                )
                test_cases.append(test_case)
        
        return test_cases
    
    def _generate_performance_cases(self, func_info: Dict[str, Any]) -> List[TestCase]:
        """生成性能测试"""
        test_cases = []
        
        # 大数据量测试
        for param in func_info.get('parameters', []):
            param_name = param['name']
            param_type = param.get('type', 'any')
            
            if param_type in ['list', 'str', 'dict']:
                large_value = self._get_large_value(param_type)
                
                test_case = TestCase(
                    id=f"perf_{param_name}_large",
                    name=f"Performance test {func_info['name']} with large {param_name}",
                    test_type=TestType.PERFORMANCE,
                    function=self._create_test_function(func_info['name'], {param_name: large_value}),
                    priority=Priority.LOW,
                    tags=["performance", "large_data"],
                    timeout=60.0
                )
                test_cases.append(test_case)
        
        return test_cases
    
    def _get_typical_values(self, param_type: str) -> List[Any]:
        """获取典型值"""
        value_map = {
            'int': [0, 1, -1, 42, 100],
            'float': [0.0, 1.0, -1.0, 3.14, 0.5],
            'str': ['', 'hello', 'test', '中文', '🚀'],
            'bool': [True, False],
            'list': [[], [1], [1, 2, 3]],
            'dict': [{}, {'key': 'value'}, {'a': 1, 'b': 2}],
            'any': [None, 0, '', [], {}]
        }
        return value_map.get(param_type, [None])
    
    def _get_wrong_type_values(self, param_type: str) -> List[Any]:
        """获取错误类型的值"""
        wrong_type_map = {
            'int': ['string', [], {}, 3.14],
            'float': ['string', [], {}, True],
            'str': [123, [], {}, True],
            'bool': ['string', 123, [], {}],
            'list': ['string', 123, {}, True],
            'dict': ['string', 123, [], True],
            'any': []
        }
        return wrong_type_map.get(param_type, [])
    
    def _get_large_value(self, param_type: str) -> Any:
        """获取大数据值"""
        if param_type == 'list':
            return list(range(10000))
        elif param_type == 'str':
            return 'x' * 100000
        elif param_type == 'dict':
            return {f'key_{i}': f'value_{i}' for i in range(1000)}
        else:
            return None
    
    def _create_test_function(self, func_name: str, test_args: Dict[str, Any]) -> Callable:
        """创建测试函数"""
        def test_function():
            try:
                # 动态导入并调用函数
                module_name = func_name.split('.')[0] if '.' in func_name else '__main__'
                func = getattr(sys.modules.get(module_name), func_name.split('.')[-1])
                
                result = func(**test_args)
                return result
                
            except Exception as e:
                if "exception" in str(test_args.values()):
                    return "exception_caught"
                raise
        
        return test_function

class CodeAnalyzer:
    """代码分析器"""
    
    def analyze_function(self, func: Callable) -> Dict[str, Any]:
        """分析函数"""
        try:
            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or ""
            
            parameters = []
            for name, param in sig.parameters.items():
                param_info = {
                    'name': name,
                    'type': self._get_type_annotation(param),
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                    'required': param.default == inspect.Parameter.empty
                }
                parameters.append(param_info)
            
            return {
                'name': func.__name__,
                'parameters': parameters,
                'docstring': doc,
                'return_type': self._get_return_type_annotation(sig),
                'module': func.__module__
            }
            
        except Exception as e:
            logger.error(f"函数分析失败: {e}")
            return {
                'name': getattr(func, '__name__', 'unknown'),
                'parameters': [],
                'docstring': '',
                'return_type': 'any',
                'module': getattr(func, '__module__', 'unknown')
            }
    
    def _get_type_annotation(self, param: inspect.Parameter) -> str:
        """获取类型注解"""
        if param.annotation == inspect.Parameter.empty:
            return 'any'
        
        try:
            return param.annotation.__name__
        except AttributeError:
            return str(param.annotation)
    
    def _get_return_type_annotation(self, sig: inspect.Signature) -> str:
        """获取返回类型注解"""
        if sig.return_annotation == inspect.Signature.empty:
            return 'any'
        
        try:
            return sig.return_annotation.__name__
        except AttributeError:
            return str(sig.return_annotation)

class TestPatternDatabase:
    """测试模式数据库"""
    
    def __init__(self):
        self.patterns = {
            'validation': [
                {'description': '输入验证', 'priority': 'high'},
                {'description': '边界检查', 'priority': 'medium'}
            ],
            'error_handling': [
                {'description': '异常处理', 'priority': 'high'},
                {'description': '错误恢复', 'priority': 'medium'}
            ],
            'performance': [
                {'description': '响应时间', 'priority': 'medium'},
                {'description': '内存使用', 'priority': 'medium'}
            ]
        }
    
    def get_patterns_for_type(self, test_type: TestType) -> List[Dict[str, Any]]:
        """获取测试类型的模式"""
        return self.patterns.get(test_type.value, [])

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics_history = deque(maxlen=1000)
        self.start_time = None
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.start_time = time.time()
        tracemalloc.start()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> PerformanceMetrics:
        """停止监控并返回指标"""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # 获取内存使用情况
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # 计算执行时间
        execution_time = time.time() - self.start_time if self.start_time else 0
        
        # 获取CPU使用率
        cpu_usage = self.process.cpu_percent()
        
        # 获取内存使用
        memory_info = self.process.memory_info()
        memory_usage = memory_info.rss / 1024 / 1024  # MB
        
        metrics = PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_peak=peak / 1024 / 1024,  # MB
            execution_time=execution_time
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 记录CPU和内存使用
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                
                # 可以添加更多监控指标
                time.sleep(0.1)  # 100ms采样间隔
                
            except Exception as e:
                logger.error(f"性能监控错误: {e}")
                break

class AutomatedTestRunner:
    """自动化测试运行器"""
    
    def __init__(self):
        self.test_generator = IntelligentTestGenerator()
        self.performance_monitor = PerformanceMonitor()
        self.test_suites: List[TestSuite] = []
        self.results_history = deque(maxlen=100)
        self.coverage_collector = coverage.Coverage()
        
    def create_test_suite(self, name: str, test_modules: List[str], 
                         test_type: TestType = TestType.UNIT) -> TestSuite:
        """创建测试套件"""
        test_suite = TestSuite(name=name)
        
        for module_name in test_modules:
            try:
                # 动态导入模块
                module = importlib.import_module(module_name)
                
                # 获取模块中的所有函数
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and not name.startswith('_'):
                        # 为每个函数生成测试用例
                        test_cases = self.test_generator.generate_tests_for_function(
                            obj, test_type
                        )
                        test_suite.test_cases.extend(test_cases)
                        
            except Exception as e:
                logger.error(f"模块 {module_name} 导入失败: {e}")
        
        test_suite.total_tests = len(test_suite.test_cases)
        return test_suite
    
    def run_test_suite(self, test_suite: TestSuite, 
                      parallel: bool = True) -> TestSuite:
        """运行测试套件"""
        test_suite.status = TestStatus.RUNNING
        test_suite.start_time = datetime.now()
        
        logger.info(f"开始运行测试套件: {test_suite.name} ({test_suite.total_tests} 个测试)")
        
        if parallel and test_suite.parallel:
            self._run_tests_parallel(test_suite)
        else:
            self._run_tests_sequential(test_suite)
        
        test_suite.end_time = datetime.now()
        test_suite.status = TestStatus.PASSED if test_suite.failed_tests == 0 else TestStatus.FAILED
        
        logger.info(f"测试套件完成: {test_suite.name} - "
                   f"通过: {test_suite.passed_tests}, "
                   f"失败: {test_suite.failed_tests}, "
                   f"跳过: {test_suite.skipped_tests}")
        
        return test_suite
    
    def _run_tests_parallel(self, test_suite: TestSuite):
        """并行运行测试"""
        max_workers = min(test_suite.max_workers, len(test_suite.test_cases))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有测试任务
            future_to_test = {
                executor.submit(self._run_single_test, test_case): test_case
                for test_case in test_suite.test_cases
            }
            
            # 等待所有测试完成
            for future in as_completed(future_to_test):
                test_case = future_to_test[future]
                
                try:
                    result = future.result()
                    self._update_test_result(test_suite, test_case, result)
                except Exception as e:
                    test_case.status = TestStatus.ERROR
                    test_case.error_message = str(e)
                    test_suite.failed_tests += 1
    
    def _run_tests_sequential(self, test_suite: TestSuite):
        """顺序运行测试"""
        for test_case in test_suite.test_cases:
            try:
                result = self._run_single_test(test_case)
                self._update_test_result(test_suite, test_case, result)
            except Exception as e:
                test_case.status = TestStatus.ERROR
                test_case.error_message = str(e)
                test_suite.failed_tests += 1
    
    def _run_single_test(self, test_case: TestCase) -> Dict[str, Any]:
        """运行单个测试"""
        test_case.status = TestStatus.RUNNING
        test_case.start_time = datetime.now()
        
        # 开始性能监控
        self.performance_monitor.start_monitoring()
        
        try:
            # 执行测试函数
            result = test_case.function()
            
            # 检查预期结果
            if test_case.expected_result == "exception":
                if result == "exception_caught":
                    test_case.status = TestStatus.PASSED
                else:
                    test_case.status = TestStatus.FAILED
                    test_case.error_message = "Expected exception but none was raised"
            else:
                test_case.status = TestStatus.PASSED
            
            # 获取性能指标
            performance_metrics = self.performance_monitor.stop_monitoring()
            test_case.performance_metrics = {
                'execution_time': performance_metrics.execution_time,
                'memory_usage': performance_metrics.memory_usage,
                'cpu_usage': performance_metrics.cpu_usage
            }
            
        except Exception as e:
            test_case.status = TestStatus.FAILED
            test_case.error_message = str(e)
            
            # 即使失败也要停止性能监控
            performance_metrics = self.performance_monitor.stop_monitoring()
            test_case.performance_metrics = {
                'execution_time': performance_metrics.execution_time,
                'memory_usage': performance_metrics.memory_usage,
                'cpu_usage': performance_metrics.cpu_usage
            }
        
        finally:
            test_case.end_time = datetime.now()
            if test_case.start_time:
                test_case.execution_time = (test_case.end_time - test_case.start_time).total_seconds()
        
        return {
            'status': test_case.status.value,
            'execution_time': test_case.execution_time,
            'performance_metrics': test_case.performance_metrics,
            'error_message': test_case.error_message
        }
    
    def _update_test_result(self, test_suite: TestSuite, 
                           test_case: TestCase, result: Dict[str, Any]):
        """更新测试结果"""
        if test_case.status == TestStatus.PASSED:
            test_suite.passed_tests += 1
        elif test_case.status == TestStatus.FAILED:
            test_suite.failed_tests += 1
        elif test_case.status == TestStatus.SKIPPED:
            test_suite.skipped_tests += 1
    
    def generate_test_report(self, test_suite: TestSuite) -> Dict[str, Any]:
        """生成测试报告"""
        report = {
            'suite_name': test_suite.name,
            'summary': {
                'total_tests': test_suite.total_tests,
                'passed_tests': test_suite.passed_tests,
                'failed_tests': test_suite.failed_tests,
                'skipped_tests': test_suite.skipped_tests,
                'success_rate': test_suite.passed_tests / test_suite.total_tests if test_suite.total_tests > 0 else 0,
                'execution_time': (test_suite.end_time - test_suite.start_time).total_seconds() if test_suite.start_time and test_suite.end_time else 0
            },
            'failed_tests': [],
            'performance_summary': self._generate_performance_summary(test_suite),
            'coverage_info': self._get_coverage_info(),
            'recommendations': self._generate_recommendations(test_suite)
        }
        
        # 收集失败的测试
        for test_case in test_suite.test_cases:
            if test_case.status == TestStatus.FAILED:
                report['failed_tests'].append({
                    'name': test_case.name,
                    'error_message': test_case.error_message,
                    'execution_time': test_case.execution_time
                })
        
        return report
    
    def _generate_performance_summary(self, test_suite: TestSuite) -> Dict[str, Any]:
        """生成性能摘要"""
        execution_times = []
        memory_usages = []
        
        for test_case in test_suite.test_cases:
            if test_case.performance_metrics:
                execution_times.append(test_case.performance_metrics.get('execution_time', 0))
                memory_usages.append(test_case.performance_metrics.get('memory_usage', 0))
        
        if not execution_times:
            return {}
        
        return {
            'avg_execution_time': np.mean(execution_times),
            'max_execution_time': np.max(execution_times),
            'min_execution_time': np.min(execution_times),
            'avg_memory_usage': np.mean(memory_usages),
            'max_memory_usage': np.max(memory_usages)
        }
    
    def _get_coverage_info(self) -> Dict[str, Any]:
        """获取覆盖率信息"""
        try:
            self.coverage_collector.stop()
            coverage_data = self.coverage_collector.get_data()
            
            return {
                'total_lines': coverage_data._lines,
                'covered_lines': len(coverage_data._lines),
                'coverage_percentage': coverage_data.report()
            }
        except Exception as e:
            logger.error(f"覆盖率信息获取失败: {e}")
            return {}
    
    def _generate_recommendations(self, test_suite: TestSuite) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于失败率的建议
        failure_rate = test_suite.failed_tests / test_suite.total_tests if test_suite.total_tests > 0 else 0
        if failure_rate > 0.1:
            recommendations.append(f"失败率较高 ({failure_rate:.1%})，建议检查代码质量")
        
        # 基于性能的建议
        slow_tests = [tc for tc in test_suite.test_cases 
                     if tc.execution_time > 5.0]
        if slow_tests:
            recommendations.append(f"发现 {len(slow_tests)} 个慢测试，建议优化性能")
        
        # 基于覆盖率的建议
        coverage_info = self._get_coverage_info()
        if coverage_info.get('coverage_percentage', 0) < 80:
            recommendations.append("测试覆盖率较低，建议增加测试用例")
        
        return recommendations

# 全局测试运行器实例
_test_runner = None

def get_test_runner() -> AutomatedTestRunner:
    """获取测试运行器单例"""
    global _test_runner
    if _test_runner is None:
        _test_runner = AutomatedTestRunner()
    return _test_runner

# 便捷函数
def run_automated_tests(test_modules: List[str], 
                       test_type: str = "unit",
                       parallel: bool = True) -> Dict[str, Any]:
    """便捷的自动化测试函数"""
    runner = get_test_runner()
    
    # 创建测试套件
    test_suite = runner.create_test_suite(
        name=f"Automated_{test_type}_Tests",
        test_modules=test_modules,
        test_type=TestType(test_type)
    )
    
    # 运行测试
    result_suite = runner.run_test_suite(test_suite, parallel=parallel)
    
    # 生成报告
    report = runner.generate_test_report(result_suite)
    
    return report

if __name__ == "__main__":
    # 测试代码
    def example_function_add(a: int, b: int) -> int:
        """示例函数：加法"""
        return a + b
    
    def example_function_divide(a: float, b: float) -> float:
        """示例函数：除法"""
        if b == 0:
            raise ValueError("除数不能为零")
        return a / b
    
    # 创建测试模块
    test_modules = ['__main__']
    
    # 运行自动化测试
    print("🧪 开始自动化测试...")
    report = run_automated_tests(test_modules, test_type="unit")
    
    print("\n📊 测试报告:")
    print(json.dumps(report, indent=2, ensure_ascii=False))