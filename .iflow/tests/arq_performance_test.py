#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQ推理引擎性能测试脚本
评估当前ARQ V2.0引擎的性能表现，识别瓶颈并提供优化建议
"""

import time
import json
import asyncio
import psutil
import os
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import pandas as pd

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from iflow.core.arq_v2_enhanced_engine import ARQV2EnhancedEngine, ReasoningMode, ProblemType
except ImportError as e:
    print(f"无法导入ARQ引擎: {e}")
    exit(1)

class ARQPerformanceTester:
    """ARQ推理引擎性能测试器"""
    
    def __init__(self):
        self.engine = None
        self.test_results = []
        self.memory_usage = []
        self.cpu_usage = []
        
    async def initialize_engine(self):
        """初始化ARQ引擎"""
        print("🚀 初始化ARQ推理引擎...")
        try:
            self.engine = ARQV2EnhancedEngine()
            print("✅ ARQ引擎初始化成功")
            return True
        except Exception as e:
            print(f"❌ ARQ引擎初始化失败: {e}")
            return False
    
    def generate_test_tasks(self) -> List[Dict[str, Any]]:
        """生成测试任务"""
        test_tasks = [
            {
                "task": "设计一个高性能的分布式缓存系统架构",
                "complexity": "high",
                "context": [{"type": "project_info", "content": "需要支持高并发读写"}]
            },
            {
                "task": "分析现有代码的性能瓶颈",
                "complexity": "medium", 
                "context": [{"type": "code_analysis", "content": "需要优化性能"}]
            },
            {
                "task": "创建一个简单的用户认证系统",
                "complexity": "low",
                "context": [{"type": "security", "content": "需要基本认证功能"}]
            }
        ]
        return test_tasks
    
    async def test_single_task(self, task_data: Dict[str, Any], task_id: int) -> Dict[str, Any]:
        """测试单个任务的性能"""
        print(f"\n🧪 测试任务 {task_id + 1}: {task_data['task'][:30]}...")
        
        # 记录初始资源使用
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        start_time = time.time()
        
        try:
            # 执行ARQ推理
            result = await self.engine.process_enhanced_reasoning(
                task=task_data["task"],
                context=task_data["context"],
                reasoning_mode=ReasoningMode.STRUCTURED,
                problem_type=ProblemType.ARCHITECTURE
            )
            
            end_time = time.time()
            
            # 记录资源使用
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_cpu = process.cpu_percent()
            
            execution_time = end_time - start_time
            memory_increase = final_memory - initial_memory
            
            # 构建测试结果
            test_result = {
                "task_id": task_id,
                "task_description": task_data["task"],
                "complexity": task_data["complexity"],
                "execution_time": execution_time,
                "success": result["success"],
                "compliance_score": result.get("compliance_score", 0),
                "confidence_score": result.get("confidence_score", 0),
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": memory_increase,
                "cpu_usage": final_cpu,
                "error": result.get("error", "")
            }
            
            self.test_results.append(test_result)
            
            # 输出结果
            print(f"  ✅ 执行结果: {'成功' if result['success'] else '失败'}")
            print(f"  ⏱️ 执行时间: {execution_time:.3f}秒")
            print(f"  📊 合规分数: {result.get('compliance_score', 0):.2f}")
            print(f"  🎯 置信度: {result.get('confidence_score', 0):.2f}")
            print(f"  💾 内存增长: {memory_increase:.2f}MB")
            
            return test_result
            
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            return {
                "task_id": task_id,
                "task_description": task_data["task"],
                "complexity": task_data["complexity"],
                "execution_time": 0,
                "success": False,
                "error": str(e)
            }
    
    async def run_performance_test(self, num_iterations: int = 5):
        """运行性能测试"""
        print(f"\n🚀 开始ARQ推理引擎性能测试 (迭代次数: {num_iterations})")
        print("=" * 60)
        
        if not await self.initialize_engine():
            return False
        
        test_tasks = self.generate_test_tasks()
        
        # 多次迭代测试
        for iteration in range(num_iterations):
            print(f"\n🔄 迭代 {iteration + 1}/{num_iterations}")
            print("-" * 40)
            
            for i, task in enumerate(test_tasks):
                result = await self.test_single_task(task, i)
                
                # 添加短暂延迟，避免资源竞争
                await asyncio.sleep(0.5)
        
        return True
    
    def analyze_results(self) -> Dict[str, Any]:
        """分析测试结果"""
        if not self.test_results:
            return {}
        
        # 计算统计指标
        execution_times = [r["execution_time"] for r in self.test_results if r["success"]]
        compliance_scores = [r["compliance_score"] for r in self.test_results if r["success"]]
        confidence_scores = [r["confidence_score"] for r in self.test_results if r["success"]]
        memory_increases = [r["memory_increase_mb"] for r in self.test_results]
        
        analysis = {
            "total_tasks": len(self.test_results),
            "successful_tasks": sum(1 for r in self.test_results if r["success"]),
            "success_rate": sum(1 for r in self.test_results if r["success"]) / len(self.test_results),
            "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "min_execution_time": min(execution_times) if execution_times else 0,
            "max_execution_time": max(execution_times) if execution_times else 0,
            "avg_compliance_score": sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0,
            "avg_confidence_score": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "avg_memory_increase": sum(memory_increases) / len(memory_increases) if memory_increases else 0,
            "max_memory_increase": max(memory_increases) if memory_increases else 0
        }
        
        return analysis
    
    def generate_performance_report(self) -> str:
        """生成性能报告"""
        analysis = self.analyze_results()
        
        report = f"""
ARQ推理引擎性能测试报告
{'=' * 60}

📊 基本指标:
- 总任务数: {analysis.get('total_tasks', 0)}
- 成功任务数: {analysis.get('successful_tasks', 0)}
- 成功率: {analysis.get('success_rate', 0):.2%}

⏱️ 执行性能:
- 平均执行时间: {analysis.get('avg_execution_time', 0):.3f}秒
- 最短执行时间: {analysis.get('min_execution_time', 0):.3f}秒
- 最长执行时间: {analysis.get('max_execution_time', 0):.3f}秒

🎯 质量指标:
- 平均合规分数: {analysis.get('avg_compliance_score', 0):.2f}
- 平均置信度: {analysis.get('avg_confidence_score', 0):.2f}

💾 资源使用:
- 平均内存增长: {analysis.get('avg_memory_increase', 0):.2f}MB
- 最大内存增长: {analysis.get('max_memory_increase', 0):.2f}MB

🔍 性能评估:
"""
        
        # 性能评估
        avg_time = analysis.get('avg_execution_time', 0)
        if avg_time < 2:
            report += "- ✅ 执行速度: 优秀 (< 2秒)\n"
        elif avg_time < 5:
            report += "- ⚠️ 执行速度: 一般 (2-5秒)\n"
        else:
            report += "- ❌ 执行速度: 较慢 (> 5秒)\n"
        
        success_rate = analysis.get('success_rate', 0)
        if success_rate > 0.95:
            report += "- ✅ 稳定性: 优秀 (> 95%)\n"
        elif success_rate > 0.8:
            report += "- ⚠️ 稳定性: 一般 (80-95%)\n"
        else:
            report += "- ❌ 稳定性: 需要改进 (< 80%)\n"
        
        avg_compliance = analysis.get('avg_compliance_score', 0)
        if avg_compliance > 0.9:
            report += "- ✅ 合规性: 优秀 (> 90%)\n"
        elif avg_compliance > 0.7:
            report += "- ⚠️ 合规性: 一般 (70-90%)\n"
        else:
            report += "- ❌ 合规性: 需要改进 (< 70%)\n"
        
        return report
    
    def save_results(self, filename: str = "arq_performance_results.json"):
        """保存测试结果"""
        results_data = {
            "test_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_tasks": len(self.test_results),
                "engine_version": "ARQ V2.0 Enhanced"
            },
            "test_results": self.test_results,
            "analysis": self.analyze_results()
        }
        
        results_path = PROJECT_ROOT / "iflow" / "tests" / "benchmark" / filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 测试结果已保存: {results_path}")
    
    async def cleanup(self):
        """清理资源"""
        if self.engine:
            await self.engine.cleanup()

async def main():
    """主函数"""
    print("🎯 ARQ推理引擎性能测试")
    print("=" * 60)
    
    # 创建测试器
    tester = ARQPerformanceTester()
    
    try:
        # 运行性能测试
        success = await tester.run_performance_test(num_iterations=3)
        
        if success:
            # 生成报告
            report = tester.generate_performance_report()
            print("\n" + report)
            
            # 保存结果
            tester.save_results()
            
        else:
            print("❌ 性能测试失败")
    
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
    
    finally:
        # 清理资源
        await tester.cleanup()

if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())