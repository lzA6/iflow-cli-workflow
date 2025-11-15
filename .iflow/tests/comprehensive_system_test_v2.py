"""
综合系统测试 V2.0
对所有核心模块进行全面测试和性能验证

测试范围：
1. ARQ推理引擎和意识流系统
2. Hooks系统功能验证
3. 多模型适配器性能测试
4. 安全框架强度测试
5. 系统集成测试
6. 性能基准测试
"""

import json
import sys
import os
import time
import asyncio
import threading
import unittest
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import sqlite3

# 导入测试模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ultimate_arq_consciousness_v2 import (
    UltimateARQConsciousnessSystem, 
    ConsciousnessStream,
    ARQEngine,
    EventType
)
from hooks.enhanced_comprehensive_hook_manager_v2 import (
    EnhancedHookManager,
    SmartContextSetup,
    EvolutionAnalyzer,
    HookType
)
from core.unified_multimodal_adapter_v4 import (
    UnifiedModelAdapter,
    ModelConfig,
    ModelProvider,
    ModelCapability
)
from tools.security.enhanced_quantum_security_framework_v8 import (
    ZeroTrustExecutor,
    SecurityValidator,
    SecurityContext,
    ThreatType,
    SecurityLevel
)

class ComprehensiveSystemTestSuite(unittest.TestCase):
    """综合系统测试套件"""
    
    def setUp(self):
        """测试前设置"""
        self.test_results = []
        self.start_time = time.time()
        
        # 初始化测试组件
        self.arq_system = UltimateARQConsciousnessSystem()
        self.hook_manager = EnhancedHookManager()
        self.model_adapter = UnifiedModelAdapter()
        self.security_executor = ZeroTrustExecutor()
        self.security_validator = SecurityValidator()
    
    def tearDown(self):
        """测试后清理"""
        total_time = time.time() - self.start_time
        print(f"\n测试完成，总耗时: {total_time:.2f}秒")
        print(f"测试结果: {len([r for r in self.test_results if r['success']])}/{len(self.test_results)} 通过")
    
    def record_test_result(self, test_name: str, success: bool, details: Dict[str, Any] = None):
        """记录测试结果"""
        result = {
            "test_name": test_name,
            "success": success,
            "details": details or {},
            "timestamp": time.time()
        }
        self.test_results.append(result)
        status = "✓" if success else "✗"
        print(f"{status} {test_name}: {'通过' if success else '失败'}")
    
    def test_arq_consciousness_system(self):
        """测试ARQ意识流系统"""
        try:
            # 测试意识流事件记录
            agent_id = "test_agent"
            task = "开发一个高性能的Web应用"
            
            # 记录事件
            self.arq_system.consciousness_stream.record_event(
                agent_id, EventType.DECISION.value, 
                {"task": task, "action": "start_development"}, 
                priority=8
            )
            
            # 获取上下文
            context = self.arq_system.consciousness_stream.get_context(agent_id)
            
            # 验证上下文包含预期内容
            assert "recent_events" in context
            assert "emotion_states" in context
            assert "global_context" in context
            
            # 测试ARQ处理
            is_valid, parsed_output, message = self.arq_system.process_task(agent_id, task)
            
            # 验证ARQ输出格式
            assert is_valid, f"ARQ验证失败: {message}"
            assert "rule_check" in parsed_output
            assert "next_action_plan" in parsed_output
            assert "confidence_score" in parsed_output
            
            # 测试情绪状态更新
            self.arq_system.consciousness_stream.update_emotion_state("confidence", 0.9)
            emotion_summary = self.arq_system.consciousness_stream.get_emotion_summary()
            assert "自信" in emotion_summary
            
            self.record_test_result("ARQ意识流系统", True, {
                "context_size": len(context["recent_events"]),
                "arq_validation": is_valid,
                "emotion_state": emotion_summary
            })
            
        except Exception as e:
            self.record_test_result("ARQ意识流系统", False, {"error": str(e)})
    
    def test_hooks_system(self):
        """测试Hooks系统"""
        try:
            # 测试智能上下文设置
            context_setup = SmartContextSetup()
            context_info = context_setup.main()
            
            assert "tech_stack" in context_info
            assert "recommendations" in context_info
            assert "super_thinking_mode" in context_info
            assert context_info["super_thinking_mode"] == True
            
            # 测试Hook执行
            result = self.hook_manager.execute_hook(HookType.SET_UP_ENVIRONMENT)
            
            assert result.success
            assert "hook_type" in result.output
            assert result.execution_time < 30  # 30秒内完成
            
            # 测试执行统计
            stats = self.hook_manager.get_execution_stats()
            assert "total_executions" in stats
            assert "success_rate" in stats
            
            self.record_test_result("Hooks系统", True, {
                "context_setup_success": True,
                "hook_execution_time": result.execution_time,
                "execution_stats": stats
            })
            
        except Exception as e:
            self.record_test_result("Hooks系统", False, {"error": str(e)})
    
    def test_multimodal_adapter(self):
        """测试多模型适配器"""
        try:
            # 测试模型配置
            model_configs = self.model_adapter.models
            assert len(model_configs) > 0, "至少应该有一个模型配置"
            
            # 验证模型配置完整性
            for model_name, config in model_configs.items():
                assert config.provider is not None
                assert config.model_name == model_name
                assert config.max_tokens > 0
                assert config.cost_per_1k_tokens >= 0
            
            # 测试Token预算管理
            session_id = "test_session_001"
            budget = self.model_adapter.create_token_budget(session_id, 10000, priority=7)
            
            assert budget.total_budget == 10000
            assert budget.remaining_tokens == 10000
            assert budget.priority_level == 7
            
            # 测试Token使用更新
            self.model_adapter.update_token_usage(session_id, 1000, "gpt-3.5-turbo")
            assert budget.used_tokens == 1000
            assert budget.remaining_tokens == 9000
            
            # 测试模型选择策略
            task_requirements = {
                "priority": 9,
                "complexity": "high",
                "budget_constraint": "medium"
            }
            
            optimal_model = self.model_adapter.get_optimal_model(task_requirements, session_id)
            assert optimal_model in model_configs
            
            # 测试模型状态获取
            model_status = self.model_adapter.get_model_status()
            assert len(model_status) == len(model_configs)
            
            self.record_test_result("多模型适配器", True, {
                "available_models": list(model_configs.keys()),
                "token_budget": asdict(budget),
                "optimal_model": optimal_model,
                "model_count": len(model_status)
            })
            
        except Exception as e:
            self.record_test_result("多模型适配器", False, {"error": str(e)})
    
    def test_security_framework(self):
        """测试安全框架"""
        try:
            # 创建安全上下文
            context = SecurityContext(
                session_id="security_test_001",
                user_id="test_user",
                permissions=["basic_execution"],
                ip_address="127.0.0.1",
                user_agent="test_agent",
                timestamp=time.time(),
                risk_score=0.3,
                trusted_sources=["localhost"]
            )
            
            # 测试安全输入验证
            safe_input = "这是一个安全的输入"
            validation_result = self.security_validator.validate_input(safe_input, context)
            
            assert validation_result["safe"] == True
            assert validation_result["risk_score"] == 0.0
            
            # 测试恶意输入检测
            malicious_inputs = [
                "'; DROP TABLE users; --",
                "<script>alert('xss')</script>",
                "../../../etc/passwd",
                "eval(malicious_code)",
                "rm -rf /",
                "sudo rm -rf /"
            ]
            
            detected_threats = 0
            for malicious_input in malicious_inputs:
                result = self.security_validator.validate_input(malicious_input, context)
                if not result["safe"]:
                    detected_threats += 1
            
            assert detected_threats > 0, "应该检测到至少一个威胁"
            
            # 测试安全命令执行
            safe_command = "echo 'hello world'"
            execution_result = self.security_executor.execute_safely(safe_command, context)
            
            assert execution_result["success"] == True
            assert "execution_time" in execution_result
            assert len(execution_result.get("violations", [])) == 0
            
            # 测试危险命令阻止
            dangerous_commands = [
                "rm -rf /",
                "sudo chmod 777 /etc/passwd",
                "eval('import os; os.system(\"ls\")')"
            ]
            
            blocked_commands = 0
            for dangerous_command in dangerous_commands:
                result = self.security_executor.execute_safely(dangerous_command, context)
                if not result["success"]:
                    blocked_commands += 1
            
            assert blocked_commands > 0, "应该阻止至少一个危险命令"
            
            self.record_test_result("安全框架", True, {
                "safe_input_detection": validation_result["safe"],
                "malicious_input_detection": detected_threats,
                "safe_command_execution": execution_result["success"],
                "dangerous_command_blocking": blocked_commands
            })
            
        except Exception as e:
            self.record_test_result("安全框架", False, {"error": str(e)})
    
    def test_performance_benchmarks(self):
        """测试性能基准"""
        try:
            performance_metrics = {}
            
            # 测试ARQ系统响应时间
            arq_times = []
            for i in range(10):
                start_time = time.time()
                agent_id = f"perf_test_agent_{i}"
                task = f"性能测试任务 {i}"
                is_valid, _, _ = self.arq_system.process_task(agent_id, task)
                arq_times.append(time.time() - start_time)
            
            performance_metrics["arq_response_time"] = {
                "avg": statistics.mean(arq_times),
                "min": min(arq_times),
                "max": max(arq_times),
                "p95": sorted(arq_times)[int(0.95 * len(arq_times))]
            }
            
            # 测试Hooks执行时间
            hook_times = []
            for i in range(5):
                start_time = time.time()
                result = self.hook_manager.execute_hook(HookType.USER_PROMPT_SUBMIT)
                hook_times.append(result.execution_time)
            
            performance_metrics["hook_execution_time"] = {
                "avg": statistics.mean(hook_times),
                "min": min(hook_times),
                "max": max(hook_times)
            }
            
            # 测试安全验证性能
            validation_times = []
            test_inputs = ["safe input"] * 20 + ["malicious input"] * 5
            
            for test_input in test_inputs:
                start_time = time.time()
                result = self.security_validator.validate_input(test_input, self._get_test_context())
                validation_times.append(time.time() - start_time)
            
            performance_metrics["security_validation_time"] = {
                "avg": statistics.mean(validation_times),
                "min": min(validation_times),
                "max": max(validation_times)
            }
            
            # 验证性能指标
            assert performance_metrics["arq_response_time"]["avg"] < 2.0, "ARQ响应时间过长"
            assert performance_metrics["hook_execution_time"]["avg"] < 5.0, "Hook执行时间过长"
            assert performance_metrics["security_validation_time"]["avg"] < 0.1, "安全验证时间过长"
            
            self.record_test_result("性能基准测试", True, performance_metrics)
            
        except Exception as e:
            self.record_test_result("性能基准测试", False, {"error": str(e)})
    
    def test_system_integration(self):
        """测试系统集成"""
        try:
            integration_results = {}
            
            # 模拟完整工作流
            session_id = "integration_test_001"
            
            # 1. 设置Token预算
            budget = self.model_adapter.create_token_budget(session_id, 20000, priority=8)
            integration_results["token_budget"] = asdict(budget)
            
            # 2. 执行Hook
            hook_result = self.hook_manager.execute_hook(HookType.SET_UP_ENVIRONMENT)
            integration_results["hook_execution"] = {
                "success": hook_result.success,
                "execution_time": hook_result.execution_time
            }
            
            # 3. ARQ处理任务
            agent_id = "integration_test_agent"
            task = "开发一个安全的高性能Web应用，需要集成用户认证和数据加密功能"
            
            arq_start = time.time()
            is_valid, arq_output, arq_message = self.arq_system.process_task(agent_id, task)
            arq_time = time.time() - arq_start
            
            integration_results["arq_processing"] = {
                "success": is_valid,
                "execution_time": arq_time,
                "confidence_score": arq_output.get("confidence_score", 0) if is_valid else 0
            }
            
            # 4. 安全验证
            security_result = self.security_validator.validate_input(task, self._get_test_context())
            integration_results["security_validation"] = {
                "safe": security_result["safe"],
                "risk_score": security_result["risk_score"]
            }
            
            # 5. 模型选择和Token使用
            optimal_model = self.model_adapter.get_optimal_model(
                {"priority": 8, "complexity": "high"}, session_id
            )
            
            self.model_adapter.update_token_usage(session_id, 2000, optimal_model)
            remaining_budget = budget.remaining_tokens
            
            integration_results["model_selection"] = {
                "selected_model": optimal_model,
                "remaining_tokens": remaining_budget
            }
            
            # 验证集成流程
            assert integration_results["hook_execution"]["success"], "Hook执行应该成功"
            assert integration_results["arq_processing"]["success"], "ARQ处理应该成功"
            assert integration_results["security_validation"]["safe"], "安全验证应该通过"
            assert integration_results["model_selection"]["remaining_tokens"] > 0, "应该有剩余Token"
            
            self.record_test_result("系统集成测试", True, integration_results)
            
        except Exception as e:
            self.record_test_result("系统集成测试", False, {"error": str(e)})
    
    def test_error_handling_and_resilience(self):
        """测试错误处理和容错性"""
        try:
            error_handling_results = {}
            
            # 测试ARQ系统错误处理
            try:
                # 传入无效任务
                is_valid, _, message = self.arq_system.process_task("", "")
                error_handling_results["arq_empty_task"] = {
                    "handled_gracefully": True,
                    "message": message
                }
            except Exception as e:
                error_handling_results["arq_empty_task"] = {
                    "handled_gracefully": False,
                    "error": str(e)
                }
            
            # 测试安全框架异常处理
            try:
                # 传入异常上下文
                bad_context = SecurityContext(
                    session_id=None,
                    user_id="",
                    permissions=[],
                    ip_address="",
                    user_agent="",
                    timestamp=0,
                    risk_score=2.0,  # 超出范围
                    trusted_sources=[]
                )
                result = self.security_validator.validate_input("test", bad_context)
                error_handling_results["security_bad_context"] = {
                    "handled_gracefully": True,
                    "result": result
                }
            except Exception as e:
                error_handling_results["security_bad_context"] = {
                    "handled_gracefully": False,
                    "error": str(e)
                }
            
            # 测试数据库异常处理
            try:
                # 模拟数据库操作
                self.arq_system.consciousness_stream.record_event(
                    "test_agent", "test_type", {"test": "data"}, priority=5
                )
                context = self.arq_system.consciousness_stream.get_context("test_agent")
                error_handling_results["database_operation"] = {
                    "success": True,
                    "context_size": len(context["recent_events"])
                }
            except Exception as e:
                error_handling_results["database_operation"] = {
                    "success": False,
                    "error": str(e)
                }
            
            # 验证错误处理
            graceful_errors = sum(1 for result in error_handling_results.values() 
                                if result.get("handled_gracefully", False))
            
            assert graceful_errors >= 2, "大部分错误应该被优雅处理"
            
            self.record_test_result("错误处理和容错性", True, error_handling_results)
            
        except Exception as e:
            self.record_test_result("错误处理和容错性", False, {"error": str(e)})
    
    def test_concurrent_access(self):
        """测试并发访问"""
        try:
            concurrent_results = {}
            
            # 创建多个线程同时访问系统
            threads = []
            thread_results = []
            
            def worker(thread_id):
                try:
                    # 模拟并发操作
                    agent_id = f"concurrent_agent_{thread_id}"
                    task = f"并发测试任务 {thread_id}"
                    
                    # ARQ处理
                    is_valid, _, _ = self.arq_system.process_task(agent_id, task)
                    
                    # 安全验证
                    context = self._get_test_context()
                    security_result = self.security_validator.validate_input(task, context)
                    
                    # Hook执行
                    hook_result = self.hook_manager.execute_hook(HookType.USER_PROMPT_SUBMIT)
                    
                    thread_results.append({
                        "thread_id": thread_id,
                        "arq_success": is_valid,
                        "security_safe": security_result["safe"],
                        "hook_success": hook_result.success
                    })
                    
                except Exception as e:
                    thread_results.append({
                        "thread_id": thread_id,
                        "error": str(e)
                    })
            
            # 启动10个并发线程
            for i in range(10):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join(timeout=30)  # 30秒超时
            
            # 分析结果
            success_count = sum(1 for result in thread_results 
                              if "error" not in result and result.get("arq_success", False))
            
            concurrent_results["total_threads"] = len(threads)
            concurrent_results["successful_threads"] = success_count
            concurrent_results["thread_results"] = thread_results
            
            # 验证并发性能
            assert success_count >= 8, "至少80%的并发操作应该成功"
            
            self.record_test_result("并发访问测试", True, concurrent_results)
            
        except Exception as e:
            self.record_test_result("并发访问测试", False, {"error": str(e)})
    
    def _get_test_context(self) -> SecurityContext:
        """获取测试用安全上下文"""
        return SecurityContext(
            session_id="test_session",
            user_id="test_user",
            permissions=["basic_execution"],
            ip_address="127.0.0.1",
            user_agent="test",
            timestamp=time.time(),
            risk_score=0.3,
            trusted_sources=["localhost"]
        )

class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self):
        self.test_suite = ComprehensiveSystemTestSuite()
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """运行综合基准测试"""
        print("开始综合性能基准测试...")
        
        benchmark_results = {
            "timestamp": time.time(),
            "test_results": [],
            "performance_metrics": {},
            "system_health": {}
        }
        
        # 运行各个性能测试
        test_methods = [
            ("ARQ系统性能", self._benchmark_arq_performance),
            ("Hooks系统性能", self._benchmark_hooks_performance),
            ("安全框架性能", self._benchmark_security_performance),
            ("内存使用分析", self._benchmark_memory_usage),
            ("并发性能", self._benchmark_concurrent_performance)
        ]
        
        for test_name, test_method in test_methods:
            try:
                print(f"运行 {test_name}...")
                result = test_method()
                benchmark_results["performance_metrics"][test_name] = result
                print(f"✓ {test_name} 完成")
            except Exception as e:
                print(f"✗ {test_name} 失败: {str(e)}")
                benchmark_results["performance_metrics"][test_name] = {"error": str(e)}
        
        # 计算系统健康度
        benchmark_results["system_health"] = self._calculate_system_health(
            benchmark_results["performance_metrics"]
        )
        
        return benchmark_results
    
    def _benchmark_arq_performance(self) -> Dict[str, Any]:
        """ARQ系统性能基准"""
        times = []
        success_count = 0
        
        for i in range(100):
            start_time = time.time()
            agent_id = f"benchmark_agent_{i}"
            task = f"性能基准测试任务 {i}，需要进行深度思考和超级思考模式"
            
            try:
                is_valid, _, _ = self.test_suite.arq_system.process_task(agent_id, task)
                if is_valid:
                    success_count += 1
                times.append(time.time() - start_time)
            except Exception:
                pass
        
        return {
            "total_tests": 100,
            "successful_tests": success_count,
            "success_rate": success_count / 100,
            "response_time": {
                "avg": statistics.mean(times) if times else 0,
                "min": min(times) if times else 0,
                "max": max(times) if times else 0,
                "p95": sorted(times)[int(0.95 * len(times))] if times else 0
            }
        }
    
    def _benchmark_hooks_performance(self) -> Dict[str, Any]:
        """Hooks系统性能基准"""
        times = []
        
        for hook_type in HookType:
            start_time = time.time()
            try:
                result = self.test_suite.hook_manager.execute_hook(hook_type)
                times.append(result.execution_time)
            except Exception:
                pass
        
        return {
            "hook_types_tested": len(HookType),
            "execution_times": {
                "avg": statistics.mean(times) if times else 0,
                "max": max(times) if times else 0
            }
        }
    
    def _benchmark_security_performance(self) -> Dict[str, Any]:
        """安全框架性能基准"""
        validation_times = []
        threat_detection_count = 0
        
        test_inputs = [
            "safe input",
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "normal text",
            "eval(malicious_code)"
        ] * 10
        
        context = self.test_suite._get_test_context()
        
        for test_input in test_inputs:
            start_time = time.time()
            try:
                result = self.test_suite.security_validator.validate_input(test_input, context)
                validation_times.append(time.time() - start_time)
                
                if not result["safe"]:
                    threat_detection_count += 1
            except Exception:
                pass
        
        return {
            "total_validations": len(test_inputs),
            "threats_detected": threat_detection_count,
            "detection_rate": threat_detection_count / len([i for i in test_inputs if "malicious" in i or "DROP" in i or "script" in i or "passwd" in i]),
            "validation_time": {
                "avg": statistics.mean(validation_times) if validation_times else 0,
                "max": max(validation_times) if validation_times else 0
            }
        }
    
    def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """内存使用基准"""
        import psutil
        import gc
        
        # 强制垃圾回收
        gc.collect()
        
        # 获取当前内存使用
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # 模拟内存压力测试
        large_data = []
        for i in range(1000):
            large_data.append({"id": i, "data": "x" * 1000})
        
        # 再次获取内存使用
        memory_info_after = process.memory_info()
        
        # 清理
        del large_data
        gc.collect()
        
        memory_info_final = process.memory_info()
        
        return {
            "initial_memory_mb": memory_info.rss / 1024 / 1024,
            "peak_memory_mb": memory_info_after.rss / 1024 / 1024,
            "final_memory_mb": memory_info_final.rss / 1024 / 1024,
            "memory_growth_mb": (memory_info_after.rss - memory_info.rss) / 1024 / 1024,
            "memory_cleanup_mb": (memory_info_after.rss - memory_info_final.rss) / 1024 / 1024
        }
    
    def _benchmark_concurrent_performance(self) -> Dict[str, Any]:
        """并发性能基准"""
        import threading
        import time
        
        threads = []
        results = []
        
        def worker(worker_id):
            start_time = time.time()
            try:
                # 模拟并发工作负载
                agent_id = f"concurrent_worker_{worker_id}"
                task = f"并发工作负载测试 {worker_id}"
                
                is_valid, _, _ = self.test_suite.arq_system.process_task(agent_id, task)
                context = self.test_suite._get_test_context()
                security_result = self.test_suite.security_validator.validate_input(task, context)
                hook_result = self.test_suite.hook_manager.execute_hook(HookType.USER_PROMPT_SUBMIT)
                
                results.append({
                    "worker_id": worker_id,
                    "success": is_valid and security_result["safe"] and hook_result.success,
                    "execution_time": time.time() - start_time
                })
            except Exception as e:
                results.append({
                    "worker_id": worker_id,
                    "success": False,
                    "error": str(e)
                })
        
        # 启动50个并发线程
        for i in range(50):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待完成
        for thread in threads:
            thread.join(timeout=60)
        
        success_count = sum(1 for r in results if r.get("success", False))
        total_time = max(r.get("execution_time", 0) for r in results) if results else 0
        
        return {
            "total_workers": 50,
            "successful_workers": success_count,
            "success_rate": success_count / 50,
            "total_execution_time": total_time,
            "avg_execution_time": statistics.mean([r.get("execution_time", 0) for r in results]) if results else 0
        }
    
    def _calculate_system_health(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """计算系统健康度"""
        health_score = 0.0
        max_score = 0.0
        
        # ARQ系统健康度 (权重30%)
        if "ARQ系统性能" in metrics:
            arq_metrics = metrics["ARQ系统性能"]
            arq_score = min(arq_metrics.get("success_rate", 0) * 100, 100)
            arq_score *= min(1.0, 2.0 / max(arq_metrics.get("response_time", {}).get("avg", 1), 0.1))
            health_score += arq_score * 0.3
            max_score += 100 * 0.3
        
        # Hooks系统健康度 (权重20%)
        if "Hooks系统性能" in metrics:
            hooks_metrics = metrics["Hooks系统性能"]
            hooks_score = 100 if hooks_metrics.get("hook_types_tested", 0) >= 5 else 50
            hooks_score *= min(1.0, 5.0 / max(hooks_metrics.get("execution_times", {}).get("max", 1), 0.1))
            health_score += hooks_score * 0.2
            max_score += 100 * 0.2
        
        # 安全框架健康度 (权重30%)
        if "安全框架性能" in metrics:
            security_metrics = metrics["安全框架性能"]
            detection_rate = security_metrics.get("detection_rate", 0)
            validation_time = security_metrics.get("validation_time", {}).get("avg", 1)
            
            security_score = min(detection_rate * 100, 100)
            security_score *= min(1.0, 0.1 / max(validation_time, 0.001))  # 期望在100ms内
            health_score += security_score * 0.3
            max_score += 100 * 0.3
        
        # 并发性能健康度 (权重20%)
        if "并发性能" in metrics:
            concurrent_metrics = metrics["并发性能"]
            concurrent_score = min(concurrent_metrics.get("success_rate", 0) * 100, 100)
            health_score += concurrent_score * 0.2
            max_score += 100 * 0.2
        
        normalized_health = (health_score / max_score) * 100 if max_score > 0 else 0
        
        return {
            "overall_health_score": round(normalized_health, 2),
            "health_level": self._get_health_level(normalized_health),
            "component_scores": {
                "arq_system": arq_score if "ARQ系统性能" in metrics else 0,
                "hooks_system": hooks_score if "Hooks系统性能" in metrics else 0,
                "security_framework": security_score if "安全框架性能" in metrics else 0,
                "concurrent_performance": concurrent_score if "并发性能" in metrics else 0
            }
        }
    
    def _get_health_level(self, score: float) -> str:
        """获取健康等级"""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Fair"
        elif score >= 60:
            return "Poor"
        else:
            return "Critical"

def main():
    """主入口函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # 运行性能基准测试
        benchmark = PerformanceBenchmark()
        results = benchmark.run_comprehensive_benchmark()
        
        print("\n" + "="*60)
        print("性能基准测试报告")
        print("="*60)
        
        print(f"整体健康度: {results['system_health']['overall_health_score']}% ({results['system_health']['health_level']})")
        
        print("\n性能指标:")
        for test_name, metrics in results["performance_metrics"].items():
            if "error" not in metrics:
                print(f"  {test_name}:")
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        print(f"    {key}: {value}")
                    else:
                        print(f"    {key}: {value}")
            else:
                print(f"  {test_name}: 错误 - {metrics['error']}")
        
        # 保存结果到文件
        with open("A项目/iflow/tests/benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n详细结果已保存到: A项目/iflow/tests/benchmark_results.json")
        
    else:
        # 运行单元测试
        unittest.main(verbosity=2)

if __name__ == "__main__":
    main()