#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作流引擎性能基准测试脚本
重点关注执行效率和资源管理
"""

import time
import asyncio
import psutil
import os
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import json

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from iflow.core.ultimate_workflow_engine_v6 import UltimateWorkflowEngineV6, TaskPriority, ExecutionMode
    from iflow.core.male_system import MultiAgentLearningEngine
except ImportError as e:
    print(f"无法导入工作流引擎: {e}")
    exit(1)

class WorkflowEngineBenchmark:
    """工作流引擎性能基准测试器"""
    
    def __init__(self):
        self.engine = None
        self.male_system = None
        
        # 性能指标
        self.execution_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.resource_snapshots = []
        
        # 测试配置
        self.test_configs = {
            "concurrent_tasks": [1, 2, 4, 8, 16],
            "task_complexity": ["low", "medium", "high", "extreme"],
            "priority_levels": [TaskPriority.LOW, TaskPriority.MEDIUM, TaskPriority.HIGH],
            "execution_modes": [ExecutionMode.SINGLE_STEP, ExecutionMode.ITERATIVE, ExecutionMode.AUTONOMOUS]
        }
    
    async def initialize_engine(self) -> bool:
        """初始化工作流引擎"""
        try:
            self.engine = UltimateWorkflowEngineV6()
            await self.engine.initialize()
            print("✅ 工作流引擎初始化成功")
            return True
        except Exception as e:
            print(f"❌ 工作流引擎初始化失败: {e}")
            return False
    
    def capture_resource_snapshot(self, label: str = "") -> Dict[str, Any]:
        """捕获资源快照"""
        process = psutil.Process(os.getpid())
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
            "num_fds": process.num_fds() if hasattr(process, 'num_fds') else 0
        }
        self.resource_snapshots.append(snapshot)
        return snapshot
    
    async def test_concurrent_execution_performance(self) -> Dict[str, Any]:
        """测试并发执行性能"""
        print("\n🚀 并发执行性能测试")
        print("-" * 40)
        
        results = {}
        
        for concurrent_count in self.test_configs["concurrent_tasks"]:
            print(f"  📊 测试并发数: {concurrent_count}")
            
            # 准备并发任务
            tasks = []
            for i in range(concurrent_count):
                task_input = f"任务{i+1}: 开发一个高性能的{'分布式缓存' if i % 3 == 0 else '用户认证' if i % 3 == 1 else '数据处理'}系统"
                tasks.append({
                    "user_input": task_input,
                    "priority": TaskPriority.MEDIUM,
                    "execution_mode": ExecutionMode.AUTONOMOUS,
                    "metadata": {"complexity": "medium", "test_type": "concurrent"}
                })
            
            # 捕获初始状态
            initial_snapshot = self.capture_resource_snapshot(f"Concurrent_{concurrent_count}_Start")
            start_time = time.time()
            
            try:
                # 并发执行任务
                semaphore = asyncio.Semaphore(concurrent_count)
                
                async def execute_task_with_semaphore(task_config):
                    async with semaphore:
                        return await self.engine.execute_task(**task_config)
                
                task_coroutines = [execute_task_with_semaphore(task) for task in tasks]
                results_list = await asyncio.gather(*task_coroutines, return_exceptions=True)
                
                end_time = time.time()
                final_snapshot = self.capture_resource_snapshot(f"Concurrent_{concurrent_count}_End")
                
                # 分析结果
                successful_tasks = sum(1 for r in results_list if not isinstance(r, Exception) and r.success)
                total_time = end_time - start_time
                avg_time_per_task = total_time / len(results_list)
                
                # 计算资源使用峰值
                peak_memory = max(s["memory_mb"] for s in self.resource_snapshots 
                                if s["label"].startswith(f"Concurrent_{concurrent_count}"))
                
                test_result = {
                    "concurrent_count": concurrent_count,
                    "total_tasks": len(tasks),
                    "successful_tasks": successful_tasks,
                    "success_rate": successful_tasks / len(tasks),
                    "total_execution_time": total_time,
                    "avg_time_per_task": avg_time_per_task,
                    "throughput_tasks_per_second": len(tasks) / total_time,
                    "peak_memory_mb": peak_memory,
                    "memory_increase_mb": final_snapshot["memory_mb"] - initial_snapshot["memory_mb"],
                    "cpu_usage_peak": max(s["cpu_percent"] for s in self.resource_snapshots 
                                        if s["label"].startswith(f"Concurrent_{concurrent_count}"))
                }
                
                results[concurrent_count] = test_result
                
                print(f"    ✅ 成功率: {test_result['success_rate']:.2%}")
                print(f"    ⏱️ 总耗时: {total_time:.3f}s (平均: {avg_time_per_task:.3f}s/任务)")
                print(f"    🚀 吞吐量: {test_result['throughput_tasks_per_second']:.2f} 任务/秒")
                print(f"    💾 内存峰值: {peak_memory:.2f}MB")
                
            except Exception as e:
                print(f"    ❌ 测试失败: {e}")
                results[concurrent_count] = {"error": str(e)}
        
        return results
    
    async def test_task_complexity_performance(self) -> Dict[str, Any]:
        """测试任务复杂度性能"""
        print("\n🧠 任务复杂度性能测试")
        print("-" * 40)
        
        complexity_tasks = {
            "low": "创建一个简单的Hello World程序",
            "medium": "设计一个用户管理系统，包含注册、登录功能",
            "high": "开发一个完整的电商平台，支持商品管理、订单处理、支付集成",
            "extreme": "构建一个分布式微服务架构的大型社交网络平台，支持千万级用户并发"
        }
        
        results = {}
        
        for complexity, task_description in complexity_tasks.items():
            print(f"  📋 复杂度: {complexity} - {task_description[:30]}...")
            
            initial_snapshot = self.capture_resource_snapshot(f"Complexity_{complexity}_Start")
            start_time = time.time()
            
            try:
                # 执行任务
                result = await self.engine.execute_task(
                    user_input=task_description,
                    priority=TaskPriority.HIGH,
                    execution_mode=ExecutionMode.ITERATIVE,
                    metadata={"complexity": complexity, "test_type": "complexity"}
                )
                
                end_time = time.time()
                final_snapshot = self.capture_resource_snapshot(f"Complexity_{complexity}_End")
                
                execution_time = end_time - start_time
                memory_increase = final_snapshot["memory_mb"] - initial_snapshot["memory_mb"]
                
                test_result = {
                    "complexity": complexity,
                    "success": result.success,
                    "execution_time": execution_time,
                    "confidence_score": result.confidence_score,
                    "tool_calls": len(result.tool_calls),
                    "memory_increase_mb": memory_increase,
                    "error": result.error if not result.success else ""
                }
                
                results[complexity] = test_result
                
                print(f"    ✅ 结果: {'成功' if result.success else '失败'}")
                print(f"    ⏱️ 执行时间: {execution_time:.3f}s")
                print(f"    🎯 置信度: {result.confidence_score:.2f}")
                print(f"    🛠️ 工具调用: {len(result.tool_calls)} 次")
                
            except Exception as e:
                print(f"    ❌ 测试失败: {e}")
                results[complexity] = {"error": str(e)}
        
        return results
    
    async def test_priority_scheduling_performance(self) -> Dict[str, Any]:
        """测试优先级调度性能"""
        print("\n🎯 优先级调度性能测试")
        print("-" * 40)
        
        # 创建混合优先级任务
        mixed_tasks = [
            {"priority": TaskPriority.LOW, "task": "优化代码注释和文档"},
            {"priority": TaskPriority.MEDIUM, "task": "修复已知的性能问题"},
            {"priority": TaskPriority.HIGH, "task": "解决关键的安全漏洞"},
            {"priority": TaskPriority.HIGH, "task": "处理紧急的系统故障"},
            {"priority": TaskPriority.MEDIUM, "task": "实施新的功能特性"},
            {"priority": TaskPriority.LOW, "task": "更新用户界面样式"}
        ]
        
        results = []
        
        print(f"  📋 执行 {len(mixed_tasks)} 个混合优先级任务...")
        
        initial_snapshot = self.capture_resource_snapshot("Priority_Start")
        start_time = time.time()
        
        try:
            # 并发执行任务，观察优先级调度
            task_coroutines = []
            for i, task_config in enumerate(mixed_tasks):
                task_input = f"任务{i+1}({task_config['priority'].value}): {task_config['task']}"
                task_coroutines.append(
                    self.engine.execute_task(
                        user_input=task_input,
                        priority=task_config["priority"],
                        execution_mode=ExecutionMode.AUTONOMOUS,
                        metadata={"test_type": "priority", "original_order": i}
                    )
                )
            
            # 使用as_completed观察执行顺序
            execution_order = []
            async for completed_task in asyncio.as_completed(task_coroutines):
                execution_order.append(completed_task)
            
            # 收集所有结果
            all_results = await asyncio.gather(*execution_order, return_exceptions=True)
            
            end_time = time.time()
            final_snapshot = self.capture_resource_snapshot("Priority_End")
            
            # 分析执行顺序
            successful_results = [r for r in all_results if not isinstance(r, Exception) and r.success]
            avg_execution_time = sum(r.execution_time for r in successful_results) / len(successful_results) if successful_results else 0
            
            priority_analysis = {
                "total_tasks": len(mixed_tasks),
                "successful_tasks": len(successful_results),
                "success_rate": len(successful_results) / len(mixed_tasks),
                "total_execution_time": end_time - start_time,
                "avg_execution_time": avg_execution_time,
                "memory_increase_mb": final_snapshot["memory_mb"] - initial_snapshot["memory_mb"]
            }
            
            results = priority_analysis
            
            print(f"    ✅ 成功率: {priority_analysis['success_rate']:.2%}")
            print(f"    ⏱️ 总耗时: {priority_analysis['total_execution_time']:.3f}s")
            print(f"    📊 平均执行时间: {avg_execution_time:.3f}s")
            
        except Exception as e:
            print(f"    ❌ 测试失败: {e}")
            results = {"error": str(e)}
        
        return results
    
    async def test_execution_mode_performance(self) -> Dict[str, Any]:
        """测试执行模式性能"""
        print("\n⚙️ 执行模式性能测试")
        print("-" * 40)
        
        test_task = "开发一个高性能的API服务，支持用户认证、数据存储和实时通信"
        
        results = {}
        
        for mode in self.test_configs["execution_modes"]:
            print(f"  🔄 测试模式: {mode.value}")
            
            initial_snapshot = self.capture_resource_snapshot(f"Mode_{mode.value}_Start")
            start_time = time.time()
            
            try:
                # 执行任务
                result = await self.engine.execute_task(
                    user_input=test_task,
                    priority=TaskPriority.HIGH,
                    execution_mode=mode,
                    metadata={"test_type": "execution_mode", "mode": mode.value}
                )
                
                end_time = time.time()
                final_snapshot = self.capture_resource_snapshot(f"Mode_{mode.value}_End")
                
                execution_time = end_time - start_time
                memory_increase = final_snapshot["memory_mb"] - initial_snapshot["memory_mb"]
                
                test_result = {
                    "mode": mode.value,
                    "success": result.success,
                    "execution_time": execution_time,
                    "confidence_score": result.confidence_score,
                    "tool_calls": len(result.tool_calls),
                    "iterations": result.execution_time if hasattr(result, 'execution_time') else 0,
                    "memory_increase_mb": memory_increase,
                    "error": result.error if not result.success else ""
                }
                
                results[mode.value] = test_result
                
                print(f"    ✅ 结果: {'成功' if result.success else '失败'}")
                print(f"    ⏱️ 执行时间: {execution_time:.3f}s")
                print(f"    🎯 置信度: {result.confidence_score:.2f}")
                print(f"    🔄 迭代次数: {test_result.get('iterations', 0)}")
                
            except Exception as e:
                print(f"    ❌ 测试失败: {e}")
                results[mode.value] = {"error": str(e)}
        
        return results
    
    async def test_memory_management_performance(self) -> Dict[str, Any]:
        """测试内存管理性能"""
        print("\n💾 内存管理性能测试")
        print("-" * 40)
        
        # 长时间运行测试
        test_duration = 60  # 60秒
        memory_samples = []
        
        print(f"  📊 进行 {test_duration} 秒的内存压力测试...")
        
        start_time = time.time()
        baseline_snapshot = self.capture_resource_snapshot("Memory_Test_Start")
        baseline_memory = baseline_snapshot["memory_mb"]
        
        task_count = 0
        
        async def memory_monitor():
            """内存监控协程"""
            nonlocal memory_samples
            while time.time() - start_time < test_duration:
                snapshot = self.capture_resource_snapshot("Memory_Sample")
                memory_samples.append({
                    "timestamp": snapshot["timestamp"],
                    "elapsed": time.time() - start_time,
                    "memory_mb": snapshot["memory_mb"],
                    "memory_percent": snapshot["memory_percent"],
                    "cpu_percent": snapshot["cpu_percent"]
                })
                await asyncio.sleep(1)  # 每秒采样一次
        
        async def task_generator():
            """任务生成协程"""
            nonlocal task_count
            while time.time() - start_time < test_duration:
                try:
                    task_input = f"内存测试任务{task_count}: {['分析代码性能', '优化数据库查询', '重构系统架构'][task_count % 3]}"
                    result = await self.engine.execute_task(
                        user_input=task_input,
                        priority=TaskPriority.MEDIUM,
                        execution_mode=ExecutionMode.SINGLE_STEP,
                        metadata={"test_type": "memory_stress"}
                    )
                    task_count += 1
                    await asyncio.sleep(0.5)  # 每0.5秒生成一个任务
                except Exception:
                    await asyncio.sleep(0.1)
        
        # 同时运行内存监控和任务生成
        await asyncio.gather(
            memory_monitor(),
            task_generator()
        )
        
        # 分析内存使用情况
        final_snapshot = self.capture_resource_snapshot("Memory_Test_End")
        final_memory = final_snapshot["memory_mb"]
        
        memory_usage_stats = {
            "test_duration": test_duration,
            "total_tasks": task_count,
            "baseline_memory_mb": baseline_memory,
            "final_memory_mb": final_memory,
            "peak_memory_mb": max(s["memory_mb"] for s in memory_samples),
            "memory_increase_mb": final_memory - baseline_memory,
            "memory_growth_rate": (final_memory - baseline_memory) / test_duration if test_duration > 0 else 0,
            "avg_memory_mb": sum(s["memory_mb"] for s in memory_samples) / len(memory_samples) if memory_samples else 0,
            "memory_efficiency": task_count / (final_memory - baseline_memory) if (final_memory - baseline_memory) > 0 else float('inf')
        }
        
        print(f"    📈 执行任务数: {task_count}")
        print(f"    💾 基线内存: {baseline_memory:.2f}MB")
        print(f"    💾 峰值内存: {memory_usage_stats['peak_memory_mb']:.2f}MB")
        print(f"    📊 平均内存: {memory_usage_stats['avg_memory_mb']:.2f}MB")
        print(f"    📈 内存增长率: {memory_usage_stats['memory_growth_rate']:.2f}MB/秒")
        
        return memory_usage_stats
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """运行综合性能基准测试"""
        print("🚀 工作流引擎综合性能基准测试")
        print("=" * 60)
        
        if not await self.initialize_engine():
            return {}
        
        # 捕获初始状态
        self.capture_resource_snapshot("Benchmark_Start")
        
        # 执行各项测试
        print("\n" + "="*60)
        concurrent_results = await self.test_concurrent_execution_performance()
        
        print("\n" + "="*60)
        complexity_results = await self.test_task_complexity_performance()
        
        print("\n" + "="*60)
        priority_results = await self.test_priority_scheduling_performance()
        
        print("\n" + "="*60)
        mode_results = await self.test_execution_mode_performance()
        
        print("\n" + "="*60)
        memory_results = await self.test_memory_management_performance()
        
        # 捕获最终状态
        self.capture_resource_snapshot("Benchmark_End")
        
        # 生成综合报告
        comprehensive_results = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_type": "workflow_engine_benchmark",
                "engine_version": "v6",
                "total_tests": 5
            },
            "concurrent_execution": concurrent_results,
            "task_complexity": complexity_results,
            "priority_scheduling": priority_results,
            "execution_modes": mode_results,
            "memory_management": memory_results,
            "resource_snapshots": self.resource_snapshots,
            "overall_performance": self.calculate_overall_performance(concurrent_results, complexity_results, priority_results, mode_results, memory_results)
        }
        
        return comprehensive_results
    
    def calculate_overall_performance(self, concurrent_results, complexity_results, priority_results, mode_results, memory_results) -> Dict[str, Any]:
        """计算整体性能指标"""
        overall = {
            "efficiency_score": 0,
            "scalability_score": 0,
            "reliability_score": 0,
            "resource_optimization_score": 0
        }
        
        # 计算效率分数 (基于并发性能和执行时间)
        if concurrent_results:
            max_throughput = max(r.get("throughput_tasks_per_second", 0) for r in concurrent_results.values() if isinstance(r, dict))
            overall["efficiency_score"] = min(max_throughput / 10, 1.0)  # 标准化到0-1
        
        # 计算可扩展性分数 (基于并发性能的线性度)
        if len(concurrent_results) >= 2:
            throughput_values = [r.get("throughput_tasks_per_second", 0) for r in concurrent_results.values() if isinstance(r, dict)]
            if len(throughput_values) > 1:
                # 简单的线性相关性计算
                import numpy as np
                concurrency_levels = list(concurrent_results.keys())
                correlation = np.corrcoef(concurrency_levels, throughput_values)[0, 1] if len(throughput_values) == len(concurrency_levels) else 0
                overall["scalability_score"] = max(0, correlation)
        
        # 计算可靠性分数 (基于成功率)
        success_rates = []
        if complexity_results:
            success_rates.extend([r.get("success_rate", 0) for r in complexity_results.values() if isinstance(r, dict)])
        if priority_results and isinstance(priority_results, dict):
            success_rates.append(priority_results.get("success_rate", 0))
        
        if success_rates:
            overall["reliability_score"] = sum(success_rates) / len(success_rates)
        
        # 计算资源优化分数 (基于内存使用效率)
        if memory_results and isinstance(memory_results, dict):
            memory_efficiency = memory_results.get("memory_efficiency", 0)
            overall["resource_optimization_score"] = min(memory_efficiency / 100, 1.0)  # 标准化到0-1
        
        # 计算综合性能分数
        overall["composite_score"] = sum(overall.values()) / len(overall)
        
        return overall
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """生成性能报告"""
        overall = results.get("overall_performance", {})
        
        report = f"""
工作流引擎性能基准测试报告
{'=' * 60}

🎯 综合性能评分:
- 效率分数: {overall.get('efficiency_score', 0):.2f}/1.0
- 可扩展性分数: {overall.get('scalability_score', 0):.2f}/1.0  
- 可靠性分数: {overall.get('reliability_score', 0):.2%}
- 资源优化分数: {overall.get('resource_optimization_score', 0):.2f}/1.0
- 综合性能分数: {overall.get('composite_score', 0):.2f}/1.0

📊 并发执行性能:
"""
        
        concurrent_results = results.get("concurrent_execution", {})
        if concurrent_results:
            for count, data in concurrent_results.items():
                if isinstance(data, dict) and "throughput_tasks_per_second" in data:
                    report += f"- {count}并发: {data['throughput_tasks_per_second']:.2f} 任务/秒, 成功率: {data.get('success_rate', 0):.2%}\n"
        
        report += f"""
🧠 任务复杂度性能:
"""
        complexity_results = results.get("task_complexity", {})
        if complexity_results:
            for complexity, data in complexity_results.items():
                if isinstance(data, dict) and "execution_time" in data:
                    report += f"- {complexity}: {data['execution_time']:.3f}s, 成功率: {data.get('success_rate', 0):.2%}\n"
        
        report += f"""
💾 内存管理性能:
"""
        memory_results = results.get("memory_management", {})
        if memory_results:
            report += f"- 执行任务数: {memory_results.get('total_tasks', 0)}\n"
            report += f"- 峰值内存: {memory_results.get('peak_memory_mb', 0):.2f}MB\n"
            report += f"- 内存效率: {memory_results.get('memory_efficiency', 0):.2f} 任务/MB\n"
        
        # 性能评估
        composite_score = overall.get("composite_score", 0)
        if composite_score > 0.8:
            report += "\n✅ 性能评级: 优秀 (> 0.8)\n"
            report += "💡 工作流引擎性能表现卓越，可以处理高并发和复杂任务负载。\n"
        elif composite_score > 0.6:
            report += "\n⚠️ 性能评级: 良好 (0.6-0.8)\n"
            report += "💡 性能表现良好，建议优化内存管理和并发处理策略。\n"
        else:
            report += "\n❌ 性能评级: 需要改进 (< 0.6)\n"
            report += "💡 建议重点优化执行效率、资源管理和错误处理机制。\n"
        
        return report
    
    def save_benchmark_results(self, results: Dict[str, Any], filename: str = "workflow_engine_benchmark_results.json"):
        """保存基准测试结果"""
        results_data = {
            "benchmark_metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_type": "workflow_engine_comprehensive_benchmark",
                "engine_version": "v6",
                "test_duration_minutes": (datetime.now() - datetime.fromisoformat(results.get("test_metadata", {}).get("timestamp", datetime.now().isoformat()))).total_seconds() / 60
            },
            "test_results": results,
            "performance_report": self.generate_performance_report(results)
        }
        
        results_path = PROJECT_ROOT / "iflow" / "tests" / "benchmark" / filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 基准测试结果已保存: {results_path}")
    
    async def cleanup(self):
        """清理资源"""
        if self.engine:
            await self.engine.shutdown()

async def main():
    """主函数"""
    print("🚀 工作流引擎性能基准测试启动")
    print("=" * 60)
    
    # 创建基准测试器
    benchmark = WorkflowEngineBenchmark()
    
    try:
        # 运行综合基准测试
        results = await benchmark.run_comprehensive_benchmark()
        
        if results:
            # 生成报告
            report = benchmark.generate_performance_report(results)
            print("\n" + report)
            
            # 保存结果
            benchmark.save_benchmark_results(results)
            
        else:
            print("❌ 基准测试失败")
    
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        await benchmark.cleanup()

if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())