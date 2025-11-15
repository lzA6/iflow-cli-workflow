#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意识流系统性能测试脚本
评估意识流系统的上下文管理、长期记忆和性能表现
"""

import time
import json
import asyncio
import psutil
import os
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from iflow.core.ultimate_consciousness_system import UltimateConsciousnessSystem
    from iflow.core.arq_v2_enhanced_engine import ARQV2EnhancedEngine
except ImportError as e:
    print(f"无法导入意识流系统: {e}")
    exit(1)

class ConsciousnessStreamTester:
    """意识流系统测试器"""
    
    def __init__(self):
        self.consciousness_system = None
        self.arq_engine = None
        self.test_results = []
        self.memory_usage = []
        
    async def initialize_systems(self):
        """初始化系统"""
        print("🧠 初始化意识流系统...")
        try:
            # 初始化意识流系统
            self.consciousness_system = UltimateConsciousnessSystem()
            print("✅ 意识流系统初始化成功")
            
            # 初始化ARQ引擎
            self.arq_engine = ARQV2EnhancedEngine()
            print("✅ ARQ引擎初始化成功")
            
            return True
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            return False
    
    def generate_consciousness_events(self, num_events: int = 100) -> List[Dict[str, Any]]:
        """生成意识流事件"""
        events = []
        base_time = datetime.now()
        
        for i in range(num_events):
            event = {
                "agent_id": f"agent_{i % 10}",  # 10个不同的Agent
                "event_type": ["reasoning_completed", "tool_call", "context_update", "memory_recall"][i % 4],
                "payload": {
                    "chain_id": f"chain_{i}",
                    "problem_type": ["ARCHITECTURE", "ANALYSIS", "DESIGN", "DEBUG"][i % 4],
                    "reasoning_mode": ["STRUCTURED", "CREATIVE", "ANALYTICAL"][i % 3],
                    "compliance_score": 0.7 + (i % 30) * 0.01,  # 0.7-1.0
                    "confidence_score": 0.6 + (i % 40) * 0.01,  # 0.6-1.0
                    "execution_time": 0.5 + (i % 100) * 0.1,  # 0.5-10.5秒
                    "timestamp": (base_time + timedelta(seconds=i)).isoformat()
                }
            }
            events.append(event)
        
        return events
    
    async def test_event_recording(self, num_events: int = 100) -> Dict[str, Any]:
        """测试事件记录性能"""
        print(f"\n📝 测试事件记录性能 (记录{num_events}个事件)")
        
        events = self.generate_consciousness_events(num_events)
        start_time = time.time()
        
        # 记录内存使用
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        recorded_count = 0
        errors = []
        
        for i, event in enumerate(events):
            try:
                self.consciousness_system.record_event(
                    agent_id=event["agent_id"],
                    event_type=event["event_type"],
                    payload=event["payload"]
                )
                recorded_count += 1
                
                # 每50个事件记录一次内存
                if (i + 1) % 50 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    self.memory_usage.append({
                        "event_count": i + 1,
                        "memory_mb": current_memory
                    })
                
            except Exception as e:
                errors.append(f"事件{i}: {str(e)}")
        
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024
        
        result = {
            "test_type": "event_recording",
            "total_events": num_events,
            "recorded_events": recorded_count,
            "success_rate": recorded_count / num_events,
            "execution_time": end_time - start_time,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": final_memory - initial_memory,
            "errors": errors
        }
        
        self.test_results.append(result)
        
        print(f"  ✅ 记录结果: {recorded_count}/{num_events} 成功")
        print(f"  ⏱️ 执行时间: {result['execution_time']:.3f}秒")
        print(f"  💾 内存增长: {result['memory_increase_mb']:.2f}MB")
        
        return result
    
    async def test_context_retrieval(self, lookback_windows: List[int] = [10, 50, 100]) -> Dict[str, Any]:
        """测试上下文检索性能"""
        print(f"\n🔍 测试上下文检索性能")
        
        results = []
        
        for window in lookback_windows:
            print(f"  测试时间窗口: {window}个事件")
            
            start_time = time.time()
            
            try:
                # 获取上下文
                context = self.consciousness_system.get_context(
                    agent_id="agent_0",
                    lookback_window=window
                )
                
                end_time = time.time()
                
                result = {
                    "test_type": "context_retrieval",
                    "lookback_window": window,
                    "context_size": len(context) if context else 0,
                    "execution_time": end_time - start_time,
                    "success": True
                }
                
                results.append(result)
                
                print(f"    ✅ 检索到 {len(context)} 个事件")
                print(f"    ⏱️ 耗时: {result['execution_time']:.4f}秒")
                
            except Exception as e:
                print(f"    ❌ 检索失败: {e}")
                results.append({
                    "test_type": "context_retrieval",
                    "lookback_window": window,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    async def test_memory_compression(self) -> Dict[str, Any]:
        """测试内存压缩功能"""
        print(f"\n🗜️ 测试内存压缩功能")
        
        try:
            start_time = time.time()
            
            # 触发压缩
            compression_result = self.consciousness_system._compress_enhanced_history()
            
            end_time = time.time()
            
            # 检查压缩后的状态
            compressed_size = len(self.consciousness_system.reasoning_history)
            
            result = {
                "test_type": "memory_compression",
                "execution_time": end_time - start_time,
                "success": True,
                "compressed_size": compressed_size
            }
            
            self.test_results.append(result)
            
            print(f"  ✅ 压缩完成，剩余事件数: {compressed_size}")
            print(f"  ⏱️ 压缩耗时: {result['execution_time']:.3f}秒")
            
            return result
            
        except Exception as e:
            print(f"  ❌ 压缩失败: {e}")
            return {
                "test_type": "memory_compression",
                "success": False,
                "error": str(e)
            }
    
    async def test_cross_agent_consistency(self) -> Dict[str, Any]:
        """测试跨Agent一致性"""
        print(f"\n🔄 测试跨Agent一致性")
        
        # 模拟多个Agent的交互
        agents = ["frontend-architect", "backend-architect", "security-engineer", "performance-engineer"]
        
        start_time = time.time()
        
        # 每个Agent记录一些事件
        for agent in agents:
            for i in range(25):
                self.consciousness_system.record_event(
                    agent_id=agent,
                    event_type="reasoning_completed",
                    payload={
                        "chain_id": f"{agent}_chain_{i}",
                        "problem_type": "ARCHITECTURE",
                        "reasoning_mode": "STRUCTURED",
                        "compliance_score": 0.85,
                        "confidence_score": 0.9,
                        "context_consistency": True
                    }
                )
        
        end_time = time.time()
        
        # 检查一致性
        total_events = len(self.consciousness_system.reasoning_history)
        
        result = {
            "test_type": "cross_agent_consistency",
            "num_agents": len(agents),
            "events_per_agent": 25,
            "total_events": total_events,
            "execution_time": end_time - start_time,
            "success": total_events == len(agents) * 25
        }
        
        self.test_results.append(result)
        
        print(f"  ✅ 跨Agent一致性测试完成")
        print(f"  📊 总事件数: {total_events}")
        print(f"  ⏱️ 执行时间: {result['execution_time']:.3f}秒")
        
        return result
    
    async def test_long_term_memory(self) -> Dict[str, Any]:
        """测试长期记忆功能"""
        print(f"\n🧠 测试长期记忆功能")
        
        try:
            start_time = time.time()
            
            # 检查LTM摘要
            ltm_summary = self.consciousness_system.ltm_summary
            
            end_time = time.time()
            
            result = {
                "test_type": "long_term_memory",
                "execution_time": end_time - start_time,
                "success": True,
                "ltm_entries": len(ltm_summary) if ltm_summary else 0,
                "ltm_size": len(str(ltm_summary)) if ltm_summary else 0
            }
            
            self.test_results.append(result)
            
            print(f"  ✅ LTM摘要条目数: {result['ltm_entries']}")
            print(f"  📏 LTM大小: {result['ltm_size']} 字符")
            print(f"  ⏱️ 检索耗时: {result['execution_time']:.4f}秒")
            
            return result
            
        except Exception as e:
            print(f"  ❌ LTM测试失败: {e}")
            return {
                "test_type": "long_term_memory",
                "success": False,
                "error": str(e)
            }
    
    async def run_comprehensive_test(self):
        """运行综合测试"""
        print("🧠 意识流系统综合性能测试")
        print("=" * 60)
        
        if not await self.initialize_systems():
            return False
        
        # 执行各项测试
        await self.test_event_recording(200)
        await self.test_context_retrieval([50, 100, 200])
        await self.test_cross_agent_consistency()
        await self.test_memory_compression()
        await self.test_long_term_memory()
        
        return True
    
    def analyze_results(self) -> Dict[str, Any]:
        """分析测试结果"""
        if not self.test_results:
            return {}
        
        analysis = {
            "total_tests": len(self.test_results),
            "successful_tests": sum(1 for r in self.test_results if r.get("success", False)),
            "avg_execution_time": sum(r.get("execution_time", 0) for r in self.test_results) / len(self.test_results),
            "total_memory_increase": sum(r.get("memory_increase_mb", 0) for r in self.test_results),
            "test_details": self.test_results
        }
        
        # 分析内存使用趋势
        if self.memory_usage:
            analysis["memory_trend"] = {
                "data_points": len(self.memory_usage),
                "initial_memory": self.memory_usage[0]["memory_mb"] if self.memory_usage else 0,
                "final_memory": self.memory_usage[-1]["memory_mb"] if self.memory_usage else 0,
                "max_memory": max(point["memory_mb"] for point in self.memory_usage) if self.memory_usage else 0
            }
        
        return analysis
    
    def generate_performance_report(self) -> str:
        """生成性能报告"""
        analysis = self.analyze_results()
        
        report = f"""
意识流系统性能测试报告
{'=' * 60}

📊 基本指标:
- 总测试数: {analysis.get('total_tests', 0)}
- 成功测试数: {analysis.get('successful_tests', 0)}
- 测试成功率: {analysis.get('successful_tests', 0) / analysis.get('total_tests', 1):.2%}

⏱️ 执行性能:
- 平均执行时间: {analysis.get('avg_execution_time', 0):.3f}秒
- 总内存增长: {analysis.get('total_memory_increase', 0):.2f}MB

🧠 功能评估:
"""
        
        # 详细测试结果
        for test_result in analysis.get("test_details", []):
            test_type = test_result.get("test_type", "unknown")
            success = test_result.get("success", False)
            status = "✅" if success else "❌"
            
            if test_type == "event_recording":
                report += f"- {status} 事件记录: {test_result.get('recorded_events', 0)}/{test_result.get('total_events', 0)} 成功\n"
            elif test_type == "context_retrieval":
                window = test_result.get("lookback_window", 0)
                size = test_result.get("context_size", 0)
                time_taken = test_result.get("execution_time", 0)
                report += f"- {status} 上下文检索 (窗口{window}): {size}个事件, {time_taken:.4f}秒\n"
            elif test_type == "cross_agent_consistency":
                agents = test_result.get("num_agents", 0)
                events = test_result.get("total_events", 0)
                report += f"- {status} 跨Agent一致性: {agents}个Agent, {events}个事件\n"
            elif test_type == "memory_compression":
                size = test_result.get("compressed_size", 0)
                report += f"- {status} 内存压缩: 压缩后{size}个事件\n"
            elif test_type == "long_term_memory":
                entries = test_result.get("ltm_entries", 0)
                report += f"- {status} 长期记忆: {entries}个摘要条目\n"
        
        # 性能评估
        avg_time = analysis.get('avg_execution_time', 0)
        if avg_time < 1:
            report += "\n✅ 性能评估: 优秀 (平均响应时间 < 1秒)\n"
        elif avg_time < 3:
            report += "\n⚠️ 性能评估: 良好 (平均响应时间 1-3秒)\n"
        else:
            report += "\n❌ 性能评估: 需要优化 (平均响应时间 > 3秒)\n"
        
        # 内存评估
        memory_increase = analysis.get('total_memory_increase', 0)
        if memory_increase < 50:
            report += "✅ 内存使用: 优秀 (内存增长 < 50MB)\n"
        elif memory_increase < 100:
            report += "⚠️ 内存使用: 可接受 (内存增长 50-100MB)\n"
        else:
            report += "❌ 内存使用: 需要优化 (内存增长 > 100MB)\n"
        
        return report
    
    def save_results(self, filename: str = "consciousness_performance_results.json"):
        """保存测试结果"""
        results_data = {
            "test_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_tests": len(self.test_results),
                "system_version": "Ultimate Consciousness System"
            },
            "test_results": self.test_results,
            "analysis": self.analyze_results(),
            "memory_usage": self.memory_usage
        }
        
        results_path = PROJECT_ROOT / "iflow" / "tests" / "benchmark" / filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 测试结果已保存: {results_path}")
    
    async def cleanup(self):
        """清理资源"""
        pass

async def main():
    """主函数"""
    print("🧠 意识流系统性能测试")
    print("=" * 60)
    
    # 创建测试器
    tester = ConsciousnessStreamTester()
    
    try:
        # 运行性能测试
        success = await tester.run_comprehensive_test()
        
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