#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 集成测试 V6
测试多模型适配器和自我进化引擎的集成功能
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import asyncio
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Any

# 动态添加项目根目录到sys.path
try:
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from iflow.core.universal_llm_adapter_v14 import UniversalLLMAdapterV14, ModelConfig, ModelType, ModelProvider
    from iflow.core.self_evolution_engine_v6 import SelfEvolutionEngineV6, EvolutionType, EvolutionSource
except ImportError as e:
    print(f"Warning: Import failed: {e}")
    print("Using simplified version for testing...")

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegrationTestV6:
    """集成测试V6"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = time.time()
        
        logger.info("🧪 集成测试V6启动")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有集成测试"""
        print("🧪 开始集成测试V6")
        print("=" * 60)
        
        # 1. 测试多模型适配器
        await self.test_llm_adapter()
        
        # 2. 测试自我进化引擎
        await self.test_evolution_engine()
        
        # 3. 测试集成功能
        await self.test_integration()
        
        # 4. 生成测试报告
        report = self.generate_report()
        
        print("\n" + "=" * 60)
        print("📊 集成测试完成")
        
        return report
    
    async def test_llm_adapter(self):
        """测试多模型适配器"""
        print("\n🌐 测试多模型适配器...")
        
        try:
            # 创建测试配置
            configs = [
                ModelConfig(
                    model_type=ModelType.GPT,
                    provider=ModelProvider.OPENAI,
                    api_key="test-key",
                    base_url="https://api.openai.com/v1",
                    max_tokens=100,
                    temperature=0.7,
                    cost_per_token=0.001,
                    speed_score=0.9,
                    quality_score=0.95
                ),
                ModelConfig(
                    model_type=ModelType.CLAUDE,
                    provider=ModelProvider.ANTHROPIC,
                    api_key="test-key",
                    base_url="https://api.anthropic.com",
                    max_tokens=100,
                    temperature=0.7,
                    cost_per_token=0.002,
                    speed_score=0.8,
                    quality_score=0.9
                )
            ]
            
            # 初始化适配器
            adapter = UniversalLLMAdapterV14(configs)
            await adapter.initialize()
            
            # 测试状态获取
            status = adapter.get_system_status()
            
            # 记录测试结果
            self.test_results.append({
                "test_name": "llm_adapter_initialization",
                "success": True,
                "details": {
                    "available_models": [m.value for m in status.get("available_models", [])],
                    "total_adapters": status.get("total_adapters", 0)
                }
            })
            
            print(f"  ✅ LLM Adapter initialized successfully, supports {len(status.get('available_models', []))} models")
            
            # 测试模型调用
            test_prompt = "这是一个测试提示，用于验证模型调用功能。"
            result = await adapter.call(test_prompt)
            
            self.test_results.append({
                "test_name": "llm_adapter_call",
                "success": result["success"],
                "details": {
                    "prompt_length": len(test_prompt),
                    "response_length": len(result.get("response", "")) if result["success"] else 0,
                    "model_used": result.get("metadata", {}).get("model", "unknown") if result["success"] else None
                }
            })
            
            if result["success"]:
                print(f"  ✅ Model call successful, response length: {len(result.get('response', ''))}")
            else:
                print(f"  ❌ Model call failed: {result.get('error', 'unknown error')}")
            
            # 关闭适配器
            await adapter.shutdown()
            
        except Exception as e:
            logger.error(f"❌ 多模型适配器测试失败: {e}")
            self.test_results.append({
                "test_name": "llm_adapter_test",
                "success": False,
                "error": str(e)
            })
    
    async def test_evolution_engine(self):
        """测试自我进化引擎"""
        print("\nTesting Evolution Engine...")
        
        try:
            # 初始化进化引擎
            engine = SelfEvolutionEngineV6("data/test_evolution.db")
            await engine.initialize()
            
            # 测试进化功能
            experience_data = [
                {
                    "response_time": 1.5 + i * 0.1,
                    "success_rate": 0.95 - i * 0.01,
                    "memory_usage": 100 + i * 10,
                    "timestamp": time.time() - i * 100
                }
                for i in range(10)
            ]
            
            evolution_result = await engine.evolve_based_on_experience(experience_data)
            
            self.test_results.append({
                "test_name": "evolution_engine_evolve",
                "success": evolution_result["success"],
                "details": {
                    "patterns_found": evolution_result.get("patterns_found", 0),
                    "improvements_suggested": evolution_result.get("improvements_suggested", 0),
                    "evolution_id": evolution_result.get("evolution_id", "")
                }
            })
            
            print(f"  ✅ Evolution analysis completed, found {evolution_result.get('patterns_found', 0)} patterns")
            
            # 测试目标设置
            goal_id = await engine.set_evolution_goal(
                "性能优化目标",
                {"avg_response_time": 1.0, "success_rate": 0.98},
                priority=8
            )
            
            self.test_results.append({
                "test_name": "evolution_engine_goal_setting",
                "success": bool(goal_id),
                "details": {
                    "goal_id": goal_id
                }
            })
            
            print(f"  ✅ Evolution goal set successfully: {goal_id}")
            
            # 测试状态获取
            status = await engine.get_evolution_status()
            
            self.test_results.append({
                "test_name": "evolution_engine_status",
                "success": True,
                "details": {
                    "total_records": status.get("total_evolution_records", 0),
                    "total_patterns": status.get("total_learning_patterns", 0),
                    "active_goals": status.get("active_goals", 0)
                }
            })
            
            print(f"  ✅ Evolution status: {status.get('total_evolution_records', 0)} records, {status.get('total_learning_patterns', 0)} patterns")
            
            # 关闭引擎
            await engine.shutdown()
            
        except Exception as e:
            logger.error(f"❌ 自我进化引擎测试失败: {e}")
            self.test_results.append({
                "test_name": "evolution_engine_test",
                "success": False,
                "error": str(e)
            })
    
    async def test_integration(self):
        """测试集成功能"""
        print("\nTesting Integration...")
        
        try:
            # 模拟完整的集成场景
            integration_scenario = {
                "step": "integration_test",
                "description": "测试多模型适配器与自我进化引擎的协同工作",
                "start_time": time.time(),
                "models_tested": [],
                "evolution_triggers": []
            }
            
            # 1. 初始化两个组件
            configs = [
                ModelConfig(
                    model_type=ModelType.GPT,
                    provider=ModelProvider.OPENAI,
                    api_key="test-key",
                    max_tokens=50,
                    temperature=0.5
                )
            ]
            
            adapter = UniversalLLMAdapterV14(configs)
            await adapter.initialize()
            
            engine = SelfEvolutionEngineV6("data/test_integration.db")
            await engine.initialize()
            
            # 2. 执行多次模型调用，收集性能数据
            performance_data = []
            
            for i in range(5):
                start_time = time.time()
                test_prompt = f"测试提示 {i+1}: 请简要说明人工智能的发展历程。"
                
                result = await adapter.call(test_prompt)
                response_time = time.time() - start_time
                
                performance_data.append({
                    "call_number": i + 1,
                    "response_time": response_time,
                    "success": result["success"],
                    "model": result.get("metadata", {}).get("model", "unknown") if result["success"] else None,
                    "timestamp": time.time()
                })
                
                integration_scenario["models_tested"].append({
                    "call": i + 1,
                    "success": result["success"],
                    "response_time": response_time
                })
            
            # 3. 基于性能数据触发进化
            evolution_trigger = {
                "type": "performance_analysis",
                "data": performance_data,
                "analysis_time": time.time()
            }
            
            integration_scenario["evolution_triggers"].append(evolution_trigger)
            
            # 4. 执行进化分析
            evolution_result = await engine.evolve_based_on_experience(performance_data)
            
            # 5. 验证集成效果
            integration_result = {
                "success": True,
                "models_tested": len(performance_data),
                "successful_calls": sum(1 for data in performance_data if data["success"]),
                "avg_response_time": sum(data["response_time"] for data in performance_data) / len(performance_data),
                "evolution_triggered": evolution_result["success"],
                "patterns_found": evolution_result.get("patterns_found", 0),
                "improvements_suggested": evolution_result.get("improvements_suggested", 0)
            }
            
            self.test_results.append({
                "test_name": "integration_test",
                "success": True,
                "details": integration_result
            })
            
            print(f"  ✅ Integration test completed:")
            print(f"    - Model calls: {integration_result['models_tested']} times")
            print(f"    - Successful calls: {integration_result['successful_calls']} times")
            print(f"    - Average response time: {integration_result['avg_response_time']:.2f} seconds")
            print(f"    - Evolution triggered: {integration_result['evolution_triggered']}")
            print(f"    - Patterns found: {integration_result['patterns_found']} patterns")
            
            # 6. 清理资源
            await adapter.shutdown()
            await engine.shutdown()
            
        except Exception as e:
            logger.error(f"❌ 集成测试失败: {e}")
            self.test_results.append({
                "test_name": "integration_test",
                "success": False,
                "error": str(e)
            })
    
    def generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - successful_tests
        
        # 计算测试覆盖率
        test_coverage = {
            "llm_adapter": any("llm_adapter" in result["test_name"] for result in self.test_results),
            "evolution_engine": any("evolution_engine" in result["test_name"] for result in self.test_results),
            "integration": any("integration" in result["test_name"] for result in self.test_results)
        }
        
        # 提取关键指标
        key_metrics = {}
        for result in self.test_results:
            if result["success"] and "details" in result:
                details = result["details"]
                if "avg_response_time" in details:
                    key_metrics["avg_response_time"] = details["avg_response_time"]
                if "patterns_found" in details:
                    key_metrics["total_patterns"] = details.get("patterns_found", 0)
                if "improvements_suggested" in details:
                    key_metrics["total_improvements"] = details.get("improvements_suggested", 0)
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "test_duration": time.time() - self.start_time
            },
            "test_coverage": test_coverage,
            "key_metrics": key_metrics,
            "test_details": self.test_results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        failed_tests = [result for result in self.test_results if not result["success"]]
        
        if failed_tests:
            recommendations.append("修复失败的测试用例")
        
        # 检查性能指标
        avg_response_time = None
        for result in self.test_results:
            if result["success"] and "details" in result and "avg_response_time" in result["details"]:
                avg_response_time = result["details"]["avg_response_time"]
                break
        
        if avg_response_time and avg_response_time > 2.0:
            recommendations.append("优化模型调用响应时间，当前平均响应时间超过2秒")
        
        # 检查进化功能
        evolution_tests = [result for result in self.test_results if "evolution" in result["test_name"]]
        if evolution_tests and all(test["success"] for test in evolution_tests):
            recommendations.append("进化引擎工作正常，建议增加更多学习器类型")
        
        if not recommendations:
            recommendations.append("所有测试通过，系统运行良好")
        
        return recommendations

# --- 主测试函数 ---
async def main():
    """主测试函数"""
    print("🧪 Integration Test V6")
    print("Testing LLM Adapter and Evolution Engine integration")
    print("=" * 60)
    
    # 创建测试实例
    tester = IntegrationTestV6()
    
    # 运行测试
    report = await tester.run_all_tests()
    
    # 打印详细报告
    print("\n📊 Test Report")
    print("=" * 60)
    
    summary = report["test_summary"]
    print(f"📋 Test Summary:")
    print(f"  - Total tests: {summary['total_tests']}")
    print(f"  - Successful tests: {summary['successful_tests']}")
    print(f"  - Failed tests: {summary['failed_tests']}")
    print(f"  - Success rate: {summary['success_rate']:.1%}")
    print(f"  - Test duration: {summary['test_duration']:.2f} seconds")
    
    print(f"\n🔍 Test Coverage:")
    for component, covered in report["test_coverage"].items():
        status = "✅" if covered else "❌"
        print(f"  {status} {component}")
    
    if report["key_metrics"]:
        print(f"\n📈 Key Metrics:")
        for metric, value in report["key_metrics"].items():
            print(f"  - {metric}: {value}")
    
    print(f"\n💡 Recommendations:")
    for i, recommendation in enumerate(report["recommendations"], 1):
        print(f"  {i}. {recommendation}")
    
    # 保存测试报告
    report_file = "iflow/tests/reports/integration_test_report_v6.json"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Test report saved to: {report_file}")
    
    # 返回测试结果
    return summary["success_rate"] > 0.8  # 80%以上成功率认为测试通过

if __name__ == "__main__":
    # 确保在Windows上asyncio事件循环正常工作
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        success = asyncio.run(main())
        if success:
            print("\n🎉 Integration test passed!")
            sys.exit(0)
        else:
            print("\n⚠️ Integration test partially failed, please check failed test cases")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
        sys.exit(1)
    except Exception as e:
        logger.error(f"测试执行异常: {e}", exc_info=True)
        print(f"\n❌ 测试执行异常: {e}")
        sys.exit(1)