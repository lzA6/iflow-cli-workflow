#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的性能测试运行器
避免复杂的导入问题，直接运行基础测试
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import asyncio

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class SimplePerformanceRunner:
    """简化的性能测试运行器"""
    
    def __init__(self):
        self.test_results = {}
        self.test_scripts = [
            {
                "name": "ARQ性能测试",
                "script": "iflow/tests/arq_performance_test.py",
                "timeout": 300,
                "description": "测试ARQ推理引擎的基础性能"
            },
            {
                "name": "意识流系统测试", 
                "script": "iflow/tests/consciousness_stream_test.py",
                "timeout": 300,
                "description": "测试意识流系统的上下文管理"
            },
            {
                "name": "工作流引擎基准测试",
                "script": "iflow/tests/workflow_engine_benchmark.py", 
                "timeout": 600,
                "description": "测试工作流引擎的执行效率"
            },
            {
                "name": "Hooks系统测试",
                "script": "iflow/tests/hooks_system_test.py",
                "timeout": 180,
                "description": "测试Hooks系统的完整性和效率"
            }
        ]
    
    def run_test_script(self, script_path: str, timeout: int = 300) -> Dict[str, Any]:
        """运行单个测试脚本"""
        full_path = PROJECT_ROOT / script_path
        
        if not full_path.exists():
            return {
                "success": False,
                "error": f"测试脚本不存在: {script_path}",
                "execution_time": 0,
                "output": ""
            }
        
        try:
            print(f"运行测试: {script_path}")
            start_time = time.time()
            
            # 运行测试脚本
            result = subprocess.run(
                [sys.executable, str(full_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 分析结果
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            return {
                "success": success,
                "execution_time": execution_time,
                "returncode": result.returncode,
                "output": output,
                "error": result.stderr if not success else ""
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"测试超时 ({timeout} 秒)",
                "execution_time": timeout,
                "output": ""
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": 0,
                "output": ""
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("简化性能测试运行器启动")
        print("=" * 60)
        
        all_results = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "runner_type": "simple_performance_runner",
                "test_count": len(self.test_scripts)
            },
            "test_results": {},
            "summary": {},
            "recommendations": []
        }
        
        total_start_time = time.time()
        
        # 依次运行每个测试
        for i, test_config in enumerate(self.test_scripts):
            print(f"\n{'='*60}")
            print(f"测试 {i+1}/{len(self.test_scripts)}: {test_config['name']}")
            print(f"描述: {test_config['description']}")
            print(f"脚本: {test_config['script']}")
            print(f"超时: {test_config['timeout']} 秒")
            print(f"{'='*60}")
            
            # 同步运行测试（避免异步导入问题）
            result = self.run_test_script(test_config["script"], test_config["timeout"])
            all_results["test_results"][test_config["name"]] = result
            
            # 显示结果
            status = "成功" if result["success"] else "失败"
            print(f"结果: {status}")
            print(f"执行时间: {result['execution_time']:.2f}秒")
            
            if not result["success"]:
                print(f"错误: {result.get('error', '未知错误')}")
                if result.get("returncode", 0) != 0:
                    print(f"返回码: {result['returncode']}")
        
        total_end_time = time.time()
        total_execution_time = total_end_time - total_start_time
        
        # 生成摘要
        successful_tests = sum(1 for r in all_results["test_results"].values() if r["success"])
        total_tests = len(all_results["test_results"])
        
        all_results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "total_execution_time": total_execution_time,
            "average_execution_time": sum(r["execution_time"] for r in all_results["test_results"].values()) / total_tests if total_tests > 0 else 0,
            "fastest_test": "",
            "slowest_test": ""
        }
        
        # 找出最快和最慢的测试
        if all_results["test_results"]:
            fastest = min(all_results["test_results"].items(), key=lambda x: x[1]["execution_time"])
            slowest = max(all_results["test_results"].items(), key=lambda x: x[1]["execution_time"])
            all_results["summary"]["fastest_test"] = fastest[0]
            all_results["summary"]["slowest_test"] = slowest[0]
        
        # 生成优化建议
        all_results["recommendations"] = self.generate_recommendations(all_results["test_results"])
        
        # 保存结果
        self.save_results(all_results)
        
        # 显示最终报告
        self.display_final_report(all_results)
        
        return all_results
    
    def generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        failed_tests = [name for name, result in test_results.items() if not result["success"]]
        successful_tests = [name for name, result in test_results.items() if result["success"]]
        
        if failed_tests:
            recommendations.append(f"🔧 {len(failed_tests)} 个测试失败，需要检查相关模块")
            
            for test_name in failed_tests:
                if "ARQ" in test_name:
                    recommendations.append("优化ARQ推理引擎的导入和初始化逻辑")
                elif "consciousness" in test_name:
                    recommendations.append("检查意识流系统的依赖和配置")
                elif "workflow" in test_name:
                    recommendations.append("优化工作流引擎的并发处理能力")
                elif "Hooks" in test_name:
                    recommendations.append("清理Hooks系统的重复文件和配置冲突")
        
        # 基于执行时间的建议
        total_time = sum(result["execution_time"] for result in test_results.values())
        if total_time > 600:  # 超过10分钟
            recommendations.append("⏱️ 总执行时间较长，建议优化测试脚本性能")
        elif total_time > 300:  # 超过5分钟
            recommendations.append("⏰ 总执行时间偏长，可以考虑并行执行")
        
        # 基于成功率的建议
        success_rate = len(successful_tests) / len(test_results) if test_results else 0
        if success_rate < 0.5:
            recommendations.append("📊 成功率较低，建议优先修复基础功能")
        elif success_rate < 0.8:
            recommendations.append("📈 部分测试失败，需要针对性优化")
        else:
            recommendations.append("测试成功率良好，系统基础稳定")
        
        return recommendations
    
    def save_results(self, results: Dict[str, Any]):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_performance_results_{timestamp}.json"
        
        results_path = PROJECT_ROOT / "iflow" / "tests" / "benchmark" / filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n测试结果已保存: {results_path}")
    
    def display_final_report(self, results: Dict[str, Any]):
        """显示最终报告"""
        summary = results["summary"]
        
        print(f"\n{'='*60}")
        print("📊 简化性能测试最终报告")
        print(f"{'='*60}")
        
        print(f"总测试数: {summary['total_tests']}")
        print(f"成功测试数: {summary['successful_tests']}")
        print(f"成功率: {summary['success_rate']:.2%}")
        print(f"总执行时间: {summary['total_execution_time']:.2f}秒")
        print(f"平均执行时间: {summary['average_execution_time']:.2f}秒")
        print(f"最快测试: {summary['fastest_test']}")
        print(f"最慢测试: {summary['slowest_test']}")
        
        print(f"\n详细结果:")
        for test_name, result in results["test_results"].items():
            status = "成功" if result["success"] else "失败"
            print(f"- {test_name}: {status} ({result['execution_time']:.2f}s)")
            if not result["success"] and result.get("error"):
                print(f"  错误: {result['error'][:100]}...")
        
        print(f"\n优化建议:")
        for i, recommendation in enumerate(results["recommendations"], 1):
            print(f"{i}. {recommendation}")
        
        # 性能评级
        success_rate = summary["success_rate"]
        if success_rate == 1.0:
            rating = "优秀"
        elif success_rate >= 0.8:
            rating = "良好"
        elif success_rate >= 0.6:
            rating = "一般"
        else:
            rating = "需要改进"
        
        print(f"\n性能评级: {rating}")

async def main():
    """主函数"""
    runner = SimplePerformanceRunner()
    await runner.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())