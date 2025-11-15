#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合性能测试运行器
依次运行所有性能测试工具并生成统一分析报告
"""

import asyncio
import time
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import traceback

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

async def run_test_with_timeout(test_coro, test_name: str, timeout: int = 300):
    """运行测试并设置超时"""
    try:
        print(f"🚀 开始运行 {test_name}...")
        start_time = time.time()
        
        result = await asyncio.wait_for(test_coro, timeout=timeout)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✅ {test_name} 完成 (耗时: {execution_time:.2f}s)")
        return {
            "success": True,
            "result": result,
            "execution_time": execution_time,
            "error": None
        }
    
    except asyncio.TimeoutError:
        print(f"⏰ {test_name} 超时 (超过 {timeout} 秒)")
        return {
            "success": False,
            "result": None,
            "execution_time": timeout,
            "error": f"测试超时 ({timeout} 秒)"
        }
    
    except Exception as e:
        print(f"❌ {test_name} 失败: {e}")
        return {
            "success": False,
            "result": None,
            "execution_time": 0,
            "error": str(e)
        }

async def run_arq_performance_test():
    """运行ARQ性能测试"""
    from iflow.tests.arq_performance_test import ARQPerformanceTester
    
    tester = ARQPerformanceTester()
    try:
        success = await tester.run_comprehensive_test()
        await tester.cleanup()
        return success
    except Exception:
        # 如果导入失败，尝试直接运行测试脚本
        import subprocess
        result = subprocess.run([
            sys.executable, 
            str(PROJECT_ROOT / "iflow" / "tests" / "arq_performance_test.py")
        ], capture_output=True, text=True, timeout=300)
        return result.returncode == 0

async def run_consciousness_test():
    """运行意识流系统测试"""
    from iflow.tests.consciousness_stream_test import ComprehensiveSystemTester
    
    tester = ComprehensiveSystemTester()
    try:
        success = await tester.run_comprehensive_test()
        await tester.cleanup()
        return success
    except Exception:
        return False

async def run_workflow_benchmark():
    """运行工作流引擎基准测试"""
    from iflow.tests.workflow_engine_benchmark import WorkflowEngineBenchmark
    
    benchmark = WorkflowEngineBenchmark()
    try:
        results = await benchmark.run_comprehensive_benchmark()
        await benchmark.cleanup()
        return results
    except Exception:
        return {}

async def run_hooks_test():
    """运行Hooks系统测试"""
    from iflow.tests.hooks_system_test import HooksSystemTester
    
    tester = HooksSystemTester()
    try:
        success = await tester.run_comprehensive_hooks_test()
        return success
    except Exception:
        return False

async def run_comprehensive_performance_suite():
    """运行综合性能测试套件"""
    print("🚀 综合性能测试套件启动")
    print("=" * 60)
    
    # 测试配置
    tests = [
        {
            "name": "ARQ推理引擎性能测试",
            "coro": run_arq_performance_test(),
            "timeout": 300,
            "description": "测试ARQ推理引擎的性能和稳定性"
        },
        {
            "name": "意识流系统性能测试", 
            "coro": run_consciousness_test(),
            "timeout": 300,
            "description": "测试意识流系统的上下文管理和记忆性能"
        },
        {
            "name": "工作流引擎基准测试",
            "coro": run_workflow_benchmark(),
            "timeout": 600,  # 工作流测试可能需要更长时间
            "description": "测试工作流引擎的执行效率和资源管理"
        },
        {
            "name": "Hooks系统完整性测试",
            "coro": run_hooks_test(),
            "timeout": 180,
            "description": "测试Hooks系统的完整性和执行效率"
        }
    ]
    
    # 运行所有测试
    test_results = {}
    overall_start_time = time.time()
    
    for test_config in tests:
        test_name = test_config["name"]
        test_coro = test_config["coro"]
        timeout = test_config["timeout"]
        
        print(f"\n{'='*60}")
        print(f"测试: {test_name}")
        print(f"描述: {test_config['description']}")
        print(f"{'='*60}")
        
        # 运行测试
        result = await run_test_with_timeout(test_coro, test_name, timeout)
        test_results[test_name] = result
        
        # 如果测试失败，继续运行其他测试
        if not result["success"]:
            print(f"⚠️ {test_name} 失败，继续下一个测试...")
    
    overall_end_time = time.time()
    total_execution_time = overall_end_time - overall_start_time
    
    # 生成综合报告
    print(f"\n{'='*60}")
    print("📊 综合性能测试结果汇总")
    print(f"{'='*60}")
    
    # 统计结果
    successful_tests = sum(1 for result in test_results.values() if result["success"])
    total_tests = len(test_results)
    
    print(f"总测试数: {total_tests}")
    print(f"成功测试数: {successful_tests}")
    print(f"成功率: {successful_tests/total_tests:.2%}")
    print(f"总执行时间: {total_execution_time:.2f}秒")
    
    # 详细结果
    for test_name, result in test_results.items():
        status = "✅ 成功" if result["success"] else "❌ 失败"
        execution_time = result["execution_time"]
        error = result.get("error", "")
        
        print(f"\n📋 {test_name}:")
        print(f"   状态: {status}")
        print(f"   执行时间: {execution_time:.2f}秒")
        if error:
            print(f"   错误: {error}")
    
    # 保存综合结果
    await save_comprehensive_results(test_results, total_execution_time)
    
    return test_results

async def save_comprehensive_results(test_results: Dict[str, Any], total_time: float):
    """保存综合测试结果"""
    timestamp = datetime.now().isoformat()
    
    comprehensive_results = {
        "test_metadata": {
            "timestamp": timestamp,
            "test_type": "comprehensive_performance_suite",
            "total_execution_time": total_time,
            "test_count": len(test_results),
            "success_count": sum(1 for r in test_results.values() if r["success"])
        },
        "individual_test_results": test_results,
        "summary": generate_test_summary(test_results),
        "recommendations": generate_optimization_recommendations(test_results)
    }
    
    # 保存到文件
    results_path = PROJECT_ROOT / "iflow" / "tests" / "benchmark" / "comprehensive_performance_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 综合测试结果已保存: {results_path}")
    
    # 生成HTML报告
    await generate_html_report(comprehensive_results, results_path.with_suffix('.html'))

def generate_test_summary(test_results: Dict[str, Any]) -> Dict[str, Any]:
    """生成测试摘要"""
    summary = {
        "overall_success_rate": 0,
        "total_execution_time": 0,
        "fastest_test": "",
        "slowest_test": "",
        "most_reliable": [],
        "performance_issues": []
    }
    
    successful_tests = [name for name, result in test_results.items() if result["success"]]
    failed_tests = [name for name, result in test_results.items() if not result["success"]]
    
    if successful_tests:
        summary["most_reliable"] = successful_tests
    
    if failed_tests:
        summary["performance_issues"] = failed_tests
    
    # 计算平均执行时间
    total_time = sum(result["execution_time"] for result in test_results.values())
    summary["total_execution_time"] = total_time
    summary["overall_success_rate"] = len(successful_tests) / len(test_results) if test_results else 0
    
    # 找出最快和最慢的测试
    if test_results:
        fastest = min(test_results.items(), key=lambda x: x[1]["execution_time"])
        slowest = max(test_results.items(), key=lambda x: x[1]["execution_time"])
        summary["fastest_test"] = fastest[0]
        summary["slowest_test"] = slowest[0]
    
    return summary

def generate_optimization_recommendations(test_results: Dict[str, Any]) -> List[str]:
    """生成优化建议"""
    recommendations = []
    
    # 分析每个测试的结果
    for test_name, result in test_results.items():
        if not result["success"]:
            if "ARQ" in test_name:
                recommendations.append("🔧 优化ARQ推理引擎的错误处理和资源管理")
            elif "consciousness" in test_name:
                recommendations.append("🧠 改进意识流系统的内存管理和上下文压缩")
            elif "workflow" in test_name:
                recommendations.append("⚙️ 优化工作流引擎的并发处理和任务调度")
            elif "hooks" in test_name:
                recommendations.append("🪝 清理和优化Hooks系统的配置和执行效率")
    
    # 基于成功率的建议
    success_rate = sum(1 for r in test_results.values() if r["success"]) / len(test_results)
    if success_rate < 0.8:
        recommendations.append("📊 整体成功率较低，建议进行全面的错误处理优化")
    elif success_rate < 1.0:
        recommendations.append("📈 部分测试失败，需要针对性优化")
    else:
        recommendations.append("🎉 所有测试通过，系统性能表现良好")
    
    # 基于执行时间的建议
    total_time = sum(result["execution_time"] for result in test_results.values())
    if total_time > 600:  # 超过10分钟
        recommendations.append("⏱️ 测试执行时间较长，建议优化性能瓶颈")
    elif total_time > 300:  # 超过5分钟
        recommendations.append("⏰ 测试执行时间偏长，可以考虑并行执行优化")
    
    return list(set(recommendations))  # 去重

async def generate_html_report(results: Dict[str, Any], output_path: Path):
    """生成HTML格式的测试报告"""
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>综合性能测试报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .summary-card h3 {{ margin: 0 0 10px 0; font-size: 2em; }}
        .summary-card p {{ margin: 0; font-size: 1.1em; opacity: 0.9; }}
        .test-result {{ margin-bottom: 20px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
        .test-header {{ background: #f8f9fa; padding: 15px; font-weight: bold; border-bottom: 1px solid #ddd; }}
        .test-content {{ padding: 15px; }}
        .success {{ color: #28a745; }}
        .failure {{ color: #dc3545; }}
        .recommendations {{ background: #e9ecef; padding: 20px; border-radius: 8px; margin-top: 20px; }}
        .recommendations h3 {{ margin-top: 0; }}
        .recommendation-item {{ margin: 10px 0; padding: 10px; background: white; border-radius: 4px; border-left: 4px solid #007bff; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 综合性能测试报告</h1>
            <p>生成时间: {results['test_metadata']['timestamp']}</p>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <h3>{results['summary']['overall_success_rate']:.0%}</h3>
                <p>整体成功率</p>
            </div>
            <div class="summary-card">
                <h3>{results['test_metadata']['test_count']}</h3>
                <p>总测试数</p>
            </div>
            <div class="summary-card">
                <h3>{results['summary']['total_execution_time']:.1f}s</h3>
                <p>总执行时间</p>
            </div>
            <div class="summary-card">
                <h3>{results['test_metadata']['success_count']}</h3>
                <p>成功测试</p>
            </div>
        </div>
        
        <h2>📋 测试结果详情</h2>
"""
    
    # 添加每个测试的详细结果
    for test_name, result in results['individual_test_results'].items():
        status_class = "success" if result['success'] else "failure"
        status_text = "✅ 成功" if result['success'] else "❌ 失败"
        
        html_content += f"""
        <div class="test-result">
            <div class="test-header">{test_name}</div>
            <div class="test-content">
                <p><strong>状态:</strong> <span class="{status_class}">{status_text}</span></p>
                <p><strong>执行时间:</strong> {result['execution_time']:.2f}秒</p>
                {f'<p><strong>错误信息:</strong> {result["error"]}</p>' if not result['success'] and result['error'] else ''}
            </div>
        </div>
        """
    
    # 添加优化建议
    html_content += f"""
        <div class="recommendations">
            <h3>💡 优化建议</h3>
    """
    
    for recommendation in results['recommendations']:
        html_content += f'<div class="recommendation-item">{recommendation}</div>'
    
    html_content += """
        </div>
        
        <div style="text-align: center; margin-top: 30px; color: #666; font-size: 0.9em;">
            <p>📊 报告生成完成</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 HTML报告已生成: {output_path}")

async def main():
    """主函数"""
    print("🎯 开始运行综合性能测试套件")
    print("这将依次测试ARQ推理引擎、意识流系统、工作流引擎和Hooks系统")
    print("预计总耗时: 15-20分钟")
    print("=" * 60)
    
    try:
        # 运行综合测试套件
        results = await run_comprehensive_performance_suite()
        
        # 显示最终统计
        successful = sum(1 for r in results.values() if r["success"])
        total = len(results)
        
        print(f"\n🎉 综合性能测试完成!")
        print(f"成功: {successful}/{total} ({successful/total:.1%})")
        
        if successful == total:
            print("🎊 所有测试通过！系统性能表现优秀。")
        else:
            print("⚠️ 部分测试失败，建议查看详细报告进行优化。")
    
    except Exception as e:
        print(f"❌ 综合测试过程中出现错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # 运行综合测试
    asyncio.run(main())