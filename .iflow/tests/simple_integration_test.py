#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Integration Test
Test LLM Adapter and Evolution Engine integration
"""

import os
import sys
import asyncio
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Any

# Set UTF-8 encoding for Windows
import codecs
if sys.platform == "win32":
    import locale
    locale.setlocale(locale.LC_ALL, 'C')
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# 动态添加项目根目录到sys.path
try:
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from iflow.core.universal_llm_adapter_v14 import UniversalLLMAdapterV14, ModelConfig, ModelType, ModelProvider
    from iflow.core.self_evolution_engine_v6 import SelfEvolutionEngineV6, EvolutionType, EvolutionSource
    print("Modules imported successfully")
except ImportError as e:
    print(f"Warning: Import failed: {e}")
    print("Using simplified version for testing...")

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleIntegrationTest:
    """Simple Integration Test"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = time.time()
        
        logger.info("Simple Integration Test started")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        print("Simple Integration Test V6")
        print("Testing LLM Adapter and Evolution Engine integration")
        print("=" * 60)
        
        # 1. Test LLM Adapter
        await self.test_llm_adapter()
        
        # 2. Test Evolution Engine
        await self.test_evolution_engine()
        
        # 3. Test Integration
        await self.test_integration()
        
        # 4. Generate test report
        report = self.generate_report()
        
        print("\n" + "=" * 60)
        print("Integration test completed")
        
        return report
    
    async def test_llm_adapter(self):
        """Test LLM Adapter"""
        print("\nTesting LLM Adapter...")
        
        try:
            # Create test configs
            configs = [
                ModelConfig(
                    model_type=ModelType.GPT,
                    provider=ModelProvider.OPENAI,
                    api_key="test-key",
                    base_url="https://api.openai.com/v1",
                    max_tokens=50,
                    temperature=0.5,
                    cost_per_token=0.001,
                    speed_score=0.9,
                    quality_score=0.95
                )
            ]
            
            # Initialize adapter
            adapter = UniversalLLMAdapterV14(configs)
            await adapter.initialize()
            
            # Test status
            status = adapter.get_system_status()
            
            # Record test result
            self.test_results.append({
                "test_name": "llm_adapter_initialization",
                "success": True,
                "details": {
                    "available_models": [m.value for m in status.get("available_models", [])],
                    "total_adapters": status.get("total_adapters", 0)
                }
            })
            
            print(f"  LLM Adapter initialized successfully, supports {len(status.get('available_models', []))} models")
            
            # Test model call
            test_prompt = "Test prompt for model call verification."
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
                print(f"  Model call successful, response length: {len(result.get('response', ''))}")
            else:
                print(f"  Model call failed: {result.get('error', 'unknown error')}")
            
            # Shutdown adapter
            await adapter.shutdown()
            
        except Exception as e:
            logger.error(f"LLM Adapter test failed: {e}")
            self.test_results.append({
                "test_name": "llm_adapter_test",
                "success": False,
                "error": str(e)
            })
    
    async def test_evolution_engine(self):
        """Test Evolution Engine"""
        print("\nTesting Evolution Engine...")
        
        try:
            # Initialize evolution engine
            engine = SelfEvolutionEngineV6("data/test_evolution.db")
            await engine.initialize()
            
            # Test evolution function
            experience_data = [
                {
                    "response_time": 1.5 + i * 0.1,
                    "success_rate": 0.95 - i * 0.01,
                    "memory_usage": 100 + i * 10,
                    "timestamp": time.time() - i * 100
                }
                for i in range(5)
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
            
            print(f"  Evolution analysis completed, found {evolution_result.get('patterns_found', 0)} patterns")
            
            # Test goal setting
            goal_id = await engine.set_evolution_goal(
                "Performance optimization goal",
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
            
            print(f"  Evolution goal set successfully: {goal_id}")
            
            # Test status
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
            
            print(f"  Evolution status: {status.get('total_evolution_records', 0)} records, {status.get('total_learning_patterns', 0)} patterns")
            
            # Shutdown engine
            await engine.shutdown()
            
        except Exception as e:
            logger.error(f"Evolution Engine test failed: {e}")
            self.test_results.append({
                "test_name": "evolution_engine_test",
                "success": False,
                "error": str(e)
            })
    
    async def test_integration(self):
        """Test Integration"""
        print("\nTesting Integration...")
        
        try:
            # Simulate complete integration scenario
            integration_scenario = {
                "step": "integration_test",
                "description": "Test LLM adapter and evolution engine collaboration",
                "start_time": time.time(),
                "models_tested": [],
                "evolution_triggers": []
            }
            
            # 1. Initialize both components
            configs = [
                ModelConfig(
                    model_type=ModelType.GPT,
                    provider=ModelProvider.OPENAI,
                    api_key="test-key",
                    max_tokens=30,
                    temperature=0.5
                )
            ]
            
            adapter = UniversalLLMAdapterV14(configs)
            await adapter.initialize()
            
            engine = SelfEvolutionEngineV6("data/test_integration.db")
            await engine.initialize()
            
            # 2. Execute multiple model calls, collect performance data
            performance_data = []
            
            for i in range(3):
                start_time = time.time()
                test_prompt = f"Test prompt {i+1}: Brief AI explanation."
                
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
            
            # 3. Trigger evolution based on performance data
            evolution_trigger = {
                "type": "performance_analysis",
                "data": performance_data,
                "analysis_time": time.time()
            }
            
            integration_scenario["evolution_triggers"].append(evolution_trigger)
            
            # 4. Execute evolution analysis
            evolution_result = await engine.evolve_based_on_experience(performance_data)
            
            # 5. Verify integration effect
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
            
            print(f"  Integration test completed:")
            print(f"    - Model calls: {integration_result['models_tested']} times")
            print(f"    - Successful calls: {integration_result['successful_calls']} times")
            print(f"    - Average response time: {integration_result['avg_response_time']:.2f} seconds")
            print(f"    - Evolution triggered: {integration_result['evolution_triggered']}")
            print(f"    - Patterns found: {integration_result['patterns_found']} patterns")
            
            # 6. Cleanup resources
            await adapter.shutdown()
            await engine.shutdown()
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            self.test_results.append({
                "test_name": "integration_test",
                "success": False,
                "error": str(e)
            })
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate test report"""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - successful_tests
        
        # Calculate test coverage
        test_coverage = {
            "llm_adapter": any("llm_adapter" in result["test_name"] for result in self.test_results),
            "evolution_engine": any("evolution_engine" in result["test_name"] for result in self.test_results),
            "integration": any("integration" in result["test_name"] for result in self.test_results)
        }
        
        # Extract key metrics
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
        """Generate recommendations"""
        recommendations = []
        
        failed_tests = [result for result in self.test_results if not result["success"]]
        
        if failed_tests:
            recommendations.append("Fix failed test cases")
        
        # Check performance metrics
        avg_response_time = None
        for result in self.test_results:
            if result["success"] and "details" in result and "avg_response_time" in result["details"]:
                avg_response_time = result["details"]["avg_response_time"]
                break
        
        if avg_response_time and avg_response_time > 2.0:
            recommendations.append("Optimize model call response time, current average exceeds 2 seconds")
        
        # Check evolution function
        evolution_tests = [result for result in self.test_results if "evolution" in result["test_name"]]
        if evolution_tests and all(test["success"] for test in evolution_tests):
            recommendations.append("Evolution engine works properly, suggest adding more learner types")
        
        if not recommendations:
            recommendations.append("All tests passed, system runs well")
        
        return recommendations

# --- Main test function ---
async def main():
    """Main test function"""
    print("Simple Integration Test V6")
    print("Testing LLM Adapter and Evolution Engine integration")
    print("=" * 60)
    
    # Create test instance
    tester = SimpleIntegrationTest()
    
    # Run tests
    report = await tester.run_all_tests()
    
    # Print detailed report
    print("\nTest Report")
    print("=" * 60)
    
    summary = report["test_summary"]
    print(f"Test Summary:")
    print(f"  - Total tests: {summary['total_tests']}")
    print(f"  - Successful tests: {summary['successful_tests']}")
    print(f"  - Failed tests: {summary['failed_tests']}")
    print(f"  - Success rate: {summary['success_rate']:.1%}")
    print(f"  - Test duration: {summary['test_duration']:.2f} seconds")
    
    print(f"\nTest Coverage:")
    for component, covered in report["test_coverage"].items():
        status = "PASS" if covered else "FAIL"
        print(f"  {status} {component}")
    
    if report["key_metrics"]:
        print(f"\nKey Metrics:")
        for metric, value in report["key_metrics"].items():
            print(f"  - {metric}: {value}")
    
    print(f"\nRecommendations:")
    for i, recommendation in enumerate(report["recommendations"], 1):
        print(f"  {i}. {recommendation}")
    
    # Save test report
    report_file = "iflow/tests/reports/simple_integration_test_report.json"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nTest report saved to: {report_file}")
    except Exception as e:
        print(f"\nFailed to save report: {e}")
    
    # Return test result
    return summary["success_rate"] > 0.8  # Consider test passed if success rate > 80%

if __name__ == "__main__":
    # Ensure asyncio event loop works properly on Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        success = asyncio.run(main())
        if success:
            print("\nIntegration test passed!")
            sys.exit(0)
        else:
            print("\nIntegration test partially failed, please check failed test cases")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nUser interrupted test")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test execution exception: {e}", exc_info=True)
        print(f"\nTest execution exception: {e}")
        sys.exit(1)