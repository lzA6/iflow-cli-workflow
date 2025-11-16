#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQ分析命令 - 统一入口
==================

这是/arq-analysis命令的统一入口点，自动选择最适合的版本

作者: AI架构师团队
版本: 16.0.0
日期: 2025-11-16
"""

import sys
import os
import time
import argparse
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description="ARQ分析系统")
    parser.add_argument("--mode", choices=["lite", "optimized", "full"], default="lite", help="分析模式")
    parser.add_argument("--thinking-mode", 
                       choices=["quantum_evolution", "predictive_causal", "anti_fragile", 
                               "collective_intelligence", "innovative_creativity"],
                       default="quantum_evolution", help="思考模式（仅完整版）")
    parser.add_argument("--batch", action="store_true", help="批量分析模式")
    parser.add_argument("query", nargs="*", help="分析查询")
    
    args = parser.parse_args()
    query = " ".join(args.query) if args.query else "系统分析"
    
    # 根据模式选择工作流
    if args.mode == "lite":
        # 使用轻量版
        print("🚀 使用ARQ轻量版分析...")
        from arq_analysis_lite_v16 import ARQLiteAnalyzer
        analyzer = ARQLiteAnalyzer()
        result = analyzer.analyze(query)
        analyzer.display_results(result)
        
    elif args.mode == "optimized":
        # 使用优化版
        print("⚡ 使用ARQ优化版分析...")
        import asyncio
        from arq_analysis_workflow_v16_final import get_analyzer
        
        async def run_optimized():
            analyzer = get_analyzer()
            if args.batch:
                queries = [
                    "iFlow CLI架构分析",
                    "系统性能优化",
                    "ARQ推理引擎",
                    "REFRAG检索系统",
                    "HRRK混合检索"
                ]
                results = await analyzer.batch_analyze(queries, args.thinking_mode)
                
                print(f"\n📊 批量分析结果:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result.get('query', 'N/A')}")
                    if "performance" in result:
                        perf = result["performance"]
                        print(f"   耗时: {perf.get('analysis_time', 0):.2f}秒")
                        print(f"   内存: {perf.get('memory_used_mb', 0):.1f}MB")
            else:
                result = await analyzer.analyze(query, args.thinking_mode)
                
                print(f"\n📊 分析结果:")
                print(f"查询: {result.get('query', 'N/A')}")
                print(f"模式: {result.get('mode', 'N/A')}")
                
                if "performance" in result:
                    perf = result["performance"]
                    print(f"\n⚡ 性能指标:")
                    print(f"  分析耗时: {perf.get('analysis_time', 0):.2f}秒")
                    print(f"  内存使用: {perf.get('memory_used_mb', 0):.1f}MB")
                    print(f"  CPU使用: {perf.get('cpu_percent', 0):.1f}%")
                
                if result.get("components_status"):
                    print(f"\n🔧 组件状态:")
                    for comp, status in result["components_status"].items():
                        print(f"  {comp}: {status}")
        
        asyncio.run(run_optimized())
        
    else:
        # 使用完整版
        print("🔬 使用ARQ完整版分析...")
        import asyncio
        from arq_analysis_workflow_v16_final import ARQAnalysisWorkflowV16
        
        async def run_full():
            workflow = ARQAnalysisWorkflowV16()
            await workflow.initialize()
            result = await workflow.analyze(query, args.thinking_mode)
            print(f"\n✅ 完整版分析完成，耗时: {result['execution_time']:.2f}秒")
        
        asyncio.run(run_full())

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
        total_time = time.time() - start_time
        print(f"\n⏱️  总耗时: {total_time:.2f}秒")
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()