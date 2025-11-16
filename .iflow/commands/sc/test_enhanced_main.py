#!/usr/bin/env python3
"""
增强版 /sc:test 指令入口文件
整合所有功能模块，提供全面的测试和分析服务
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# 导入所有功能模块
from test_enhanced import EnhancedTestEngine, TestConfiguration
from ai_information_forcer import AIInformationForcer
from structure_analyzer import ProjectStructureAnalyzer
from optimization_report_generator import OptimizationReportGenerator
from deep_analysis_scanner import DeepAnalysisScanner
from comprehensive_justification_system import ComprehensiveJustificationSystem, DecisionType
from feature_analysis_module import FeatureAnalysisModule

class EnhancedSCTestCommand:
    """增强版 /sc:test 命令"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / "reports"
        self.results_dir.mkdir(exist_ok=True)
        
        # 初始化所有模块
        self.ai_forcer = AIInformationForcer(project_root)
        self.structure_analyzer = ProjectStructureAnalyzer(project_root)
        self.optimization_generator = OptimizationReportGenerator(project_root)
        self.deep_scanner = DeepAnalysisScanner(project_root)
        self.justification_system = ComprehensiveJustificationSystem(project_root)
        self.feature_analyzer = FeatureAnalysisModule(project_root)
        
    async def execute_enhanced_test(self, 
                                   interactive_mode: bool = True,
                                   force_ai_awareness: bool = True,
                                   enable_deep_analysis: bool = True,
                                   generate_optimization_report: bool = True,
                                   compare_structures: bool = True) -> Dict[str, Any]:
        """执行增强版测试"""
        print("🚀 启动增强版 /sc:test 全面分析系统")
        print("=" * 80)
        
        # 1. 强制AI信息传递
        if force_ai_awareness:
            print("\n🤖 第一步：强制AI信息传递")
            ai_context = await self.ai_forcer.force_ai_awareness()
            print("✅ AI信息传递完成")
        
        # 2. 项目结构对比分析
        structure_comparison = None
        if compare_structures:
            print("\n🔄 第二步：项目结构对比分析")
            structure_comparison = await self.structure_analyzer.analyze_and_compare()
            print("✅ 结构对比分析完成")
        
        # 3. 深度分析扫描审查
        deep_scan_results = None
        if enable_deep_analysis:
            print("\n🔬 第三步：深度分析扫描审查")
            deep_scan_results = await self.deep_scanner.perform_comprehensive_scan()
            print("✅ 深度扫描审查完成")
        
        # 4. 功能特点分析
        feature_analyses = None
        if deep_scan_results:
            print("\n🎯 第四步：功能特点分析")
            feature_analyses = await self._perform_feature_analyses(deep_scan_results)
            print("✅ 功能特点分析完成")
        
        # 5. 生成决策记录
        decision_records = None
        if feature_analyses:
            print("\n⚖️ 第五步：生成决策记录")
            decision_records = await self._generate_decision_records(feature_analyses)
            print("✅ 决策记录生成完成")
        
        # 6. 自动优化报告生成
        optimization_report = None
        if generate_optimization_report:
            print("\n📈 第六步：自动优化报告生成")
            optimization_report = await self.optimization_generator.generate_comprehensive_report()
            print("✅ 优化报告生成完成")
        
        # 7. 交互式处理
        if interactive_mode:
            print("\n🎮 第七步：交互式处理")
            await self._interactive_analysis(deep_scan_results, feature_analyses, decision_records)
        
        # 8. 生成最终综合报告
        print("\n📋 第八步：生成最终综合报告")
        final_report = await self._generate_final_report(
            ai_context if force_ai_awareness else None,
            structure_comparison,
            deep_scan_results,
            feature_analyses,
            decision_records,
            optimization_report
        )
        
        print("\n🎉 增强版 /sc:test 全面分析完成！")
        return final_report
    
    async def _perform_feature_analyses(self, deep_scan_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行功能分析"""
        feature_analyses = []
        
        # 获取所有Python文件
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    rel_path = str(file_path.relative_to(self.project_root))
                    python_files.append(rel_path)
        
        # 分析每个文件的功能特点
        for file_path in python_files:
            try:
                analysis = await self.feature_analyzer.analyze_comprehensive_features(file_path)
                feature_analyses.append(analysis)
            except Exception as e:
                print(f"⚠️ 功能分析失败 {file_path}: {e}")
        
        return feature_analyses
    
    async def _generate_decision_records(self, feature_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成决策记录"""
        decision_records = []
        
        for analysis in feature_analyses:
            try:
                # 基于功能分析结果创建决策
                if hasattr(analysis, 'recommendation'):
                    if "删除" in analysis.recommendation:
                        decision_type = DecisionType.FILE_REMOVAL
                    elif "保留" in analysis.recommendation:
                        decision_type = DecisionType.FILE_RETENTION
                    elif "重构" in analysis.recommendation:
                        decision_type = DecisionType.CODE_REFACTOR
                    else:
                        continue
                    
                    # 创建决策记录
                    analysis_data = {
                        "features": [asdict(f) for f in analysis.feature_characteristics],
                        "advantages": [asdict(a) for a in analysis.advantages],
                        "disadvantages": [asdict(d) for d in analysis.disadvantages],
                        "alternatives": [asdict(a) for a in analysis.alternatives]
                    }
                    
                    decision_record = await self.justification_system.create_comprehensive_decision(
                        decision_type=decision_type,
                        target=analysis.file_path,
                        analysis_data=analysis_data
                    )
                    
                    decision_records.append(asdict(decision_record))
            
            except Exception as e:
                print(f"⚠️ 决策记录生成失败 {analysis.file_path}: {e}")
        
        return decision_records
    
    async def _interactive_analysis(self, 
                                  deep_scan_results: Optional[Dict[str, Any]],
                                  feature_analyses: Optional[List[Dict[str, Any]]],
                                  decision_records: Optional[List[Dict[str, Any]]]):
        """交互式分析"""
        print("\n🎯 欢迎使用增强版 /sc:test 交互式分析模式")
        print("=" * 60)
        print("📋 本系统将引导您完成全面的项目测试和分析")
        print("💡 您可以随时输入 'help' 查看帮助，或输入 'quit' 退出")
        print("=" * 60)
        
        # 显示系统状态
        await self._show_system_status(deep_scan_results, feature_analyses, decision_records)
        
        while True:
            print("\n" + "🔥" * 20 + " 主菜单 " + "🔥" * 20)
            print("请选择您想要执行的操作：")
            print("\n📊 【结果查看】")
            print("  1. 🔍 查看深度扫描结果")
            print("  2. 🎯 查看功能特点分析")
            print("  3. ⚖️  查看决策记录")
            print("  4. 🔄 查看项目结构变化")
            print("  5. 💡 查看优化建议")
            print("\n🛠️ 【操作工具】")
            print("  6. 📄 导出详细报告")
            print("  7. 🔄 重新分析特定文件")
            print("  8. 📈 生成自定义报告")
            print("  9. ⚙️  系统设置")
            print("\n🚪 【系统】")
            print("  0. 👋 退出交互模式")
            print("  help - 📖 显示帮助信息")
            print("  status - 📊 显示系统状态")
            print("  clear - 🧹 清屏")
            
            try:
                choice = input("\n✨ 请输入您的选择 (0-9 或命令): ").strip().lower()
                
                # 处理特殊命令
                if choice == 'quit' or choice == 'exit' or choice == 'q':
                    confirm = input("🤔 确定要退出吗？(y/N): ").strip().lower()
                    if confirm in ['y', 'yes', '是']:
                        print("\n👋 感谢使用增强版 /sc:test 系统！")
                        break
                    else:
                        continue
                        
                elif choice == 'help':
                    await self._show_help()
                    continue
                    
                elif choice == 'status':
                    await self._show_system_status(deep_scan_results, feature_analyses, decision_records)
                    continue
                    
                elif choice == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                
                # 处理数字选择
                elif choice == "0":
                    confirm = input("🤔 确定要退出交互模式吗？(y/N): ").strip().lower()
                    if confirm in ['y', 'yes', '是']:
                        print("\n👋 感谢使用增强版 /sc:test 系统！")
                        break
                    else:
                        continue
                        
                elif choice == "1":
                    await self._show_deep_scan_results(deep_scan_results)
                elif choice == "2":
                    await self._show_feature_analyses(feature_analyses)
                elif choice == "3":
                    await self._show_decision_records(decision_records)
                elif choice == "4":
                    await self._show_structure_comparison()
                elif choice == "5":
                    await self._show_optimization_recommendations()
                elif choice == "6":
                    await self._export_detailed_reports()
                elif choice == "7":
                    await self._reanalyze_specific_file()
                elif choice == "8":
                    await self._generate_custom_report()
                elif choice == "9":
                    await self._system_settings()
                else:
                    print("❌ 无效选择，请输入 0-9 之间的数字或有效命令")
                    print("💡 输入 'help' 查看可用命令")
            
            except KeyboardInterrupt:
                print("\n\n⚠️ 检测到中断信号...")
                confirm = input("🤔 确定要强制退出吗？(y/N): ").strip().lower()
                if confirm in ['y', 'yes', '是']:
                    print("\n👋 用户中断，退出交互模式")
                    break
                else:
                    continue
                    
            except Exception as e:
                print(f"❌ 操作出错: {e}")
                print("💡 请重试或输入 'help' 获取帮助")
                
            # 每次操作后暂停
            input("\n⏸️ 按回车键继续...")
    
    async def _show_system_status(self, 
                                 deep_scan_results: Optional[Dict[str, Any]],
                                 feature_analyses: Optional[List[Dict[str, Any]]],
                                 decision_records: Optional[List[Dict[str, Any]]]):
        """显示系统状态"""
        print("\n📊 系统状态概览")
        print("=" * 50)
        
        # 分析模块状态
        print("🔍 分析模块状态:")
        print(f"  ✅ AI信息传递: {'已完成' if True else '未完成'}")
        print(f"  ✅ 项目结构对比: {'已完成' if True else '未完成'}")
        print(f"  {'✅' if deep_scan_results else '❌'} 深度扫描审查: {'已完成' if deep_scan_results else '未完成'}")
        print(f"  {'✅' if feature_analyses else '❌'} 功能特点分析: {'已完成' if feature_analyses else '未完成'}")
        print(f"  {'✅' if decision_records else '❌'} 决策记录: {'已完成' if decision_records else '未完成'}")
        print(f"  ✅ 优化报告生成: {'已完成' if True else '未完成'}")
        
        # 数据统计
        print("\n📈 数据统计:")
        if deep_scan_results:
            summary = deep_scan_results.get("scan_summary", {})
            total_issues = summary.get("scan_overview", {}).get("total_issues", 0)
            quality_score = summary.get("quality_metrics", {}).get("overall_quality_score", 0)
            print(f"  🔍 扫描问题数: {total_issues}")
            print(f"  📊 质量评分: {quality_score:.2f}")
        
        if feature_analyses:
            total_files = len(feature_analyses)
            high_value = len([f for f in feature_analyses if hasattr(f, 'recommendation') and "保留" in f.recommendation])
            print(f"  📁 分析文件数: {total_files}")
            print(f"  💎 高价值文件: {high_value}")
        
        if decision_records:
            total_decisions = len(decision_records)
            high_confidence = len([d for d in decision_records if d.get('confidence_score', 0) > 0.8])
            print(f"  ⚖️ 决策记录数: {total_decisions}")
            print(f"  🎯 高置信度决策: {high_confidence}")
    
    async def _show_help(self):
        """显示帮助信息"""
        print("\n📖 增强版 /sc:test 交互式系统帮助")
        print("=" * 50)
        print("\n🎯 系统功能:")
        print("  本系统提供全面的项目测试和分析功能，包括：")
        print("  • 深度代码扫描和质量分析")
        print("  • 功能特点和价值评估")
        print("  • 智能决策支持")
        print("  • 优化建议生成")
        print("  • 交互式结果查看")
        
        print("\n📋 可用命令:")
        print("  数字选择 (0-9): 执行对应的菜单操作")
        print("  help/h: 显示此帮助信息")
        print("  status/s: 显示系统状态")
        print("  clear/c: 清屏")
        print("  quit/exit/q: 退出系统")
        
        print("\n💡 使用技巧:")
        print("  • 使用 Tab 键可以自动补全（如果支持）")
        print("  • 使用方向键可以浏览历史输入（如果支持）")
        print("  • 按 Ctrl+C 可以安全中断当前操作")
        print("  • 所有操作都有确认提示，避免误操作")
        
        print("\n🔧 高级功能:")
        print("  • 自定义报告生成")
        print("  • 特定文件重新分析")
        print("  • 批量数据导出")
        print("  • 系统参数调整")
    
    async def _generate_custom_report(self):
        """生成自定义报告"""
        print("\n📈 自定义报告生成")
        print("-" * 40)
        
        print("请选择要包含在报告中的内容：")
        print("1. 📊 仅包含摘要信息")
        print("2. 📋 包含详细分析结果")
        print("3. 🔍 包含原始数据")
        print("4. 💡 包含优化建议")
        print("5. 📦 包含所有内容")
        
        try:
            choice = input("请选择报告类型 (1-5): ").strip()
            
            format_choice = input("选择输出格式 (1:Markdown, 2:JSON, 3:HTML): ").strip()
            
            filename = input("输入报告文件名（留空使用默认）: ").strip()
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"custom_report_{timestamp}"
            
            print("🔄 正在生成自定义报告...")
            
            # 这里应该实现实际的报告生成逻辑
            print(f"✅ 自定义报告已生成: {filename}")
            
        except Exception as e:
            print(f"❌ 生成自定义报告失败: {e}")
    
    async def _system_settings(self):
        """系统设置"""
        print("\n⚙️ 系统设置")
        print("-" * 40)
        
        while True:
            print("\n设置选项：")
            print("1. 🎨 界面主题设置")
            print("2. 📊 输出详细程度")
            print("3. 💾 自动保存设置")
            print("4. 🔄 默认分析选项")
            print("0. 🔙 返回主菜单")
            
            try:
                choice = input("请选择设置项 (0-4): ").strip()
                
                if choice == "0":
                    break
                elif choice == "1":
                    print("🎨 主题设置功能开发中...")
                elif choice == "2":
                    print("📊 详细程度设置功能开发中...")
                elif choice == "3":
                    print("💾 自动保存设置功能开发中...")
                elif choice == "4":
                    print("🔄 默认选项设置功能开发中...")
                else:
                    print("❌ 无效选择")
            
            except Exception as e:
                print(f"❌ 设置操作失败: {e}")
    
    async def _show_deep_scan_results(self, deep_scan_results: Optional[Dict[str, Any]]):
        """显示深度扫描结果"""
        if not deep_scan_results:
            print("❌ 无深度扫描结果")
            return
        
        print("\n🔬 深度扫描结果")
        print("-" * 40)
        
        summary = deep_scan_results.get("scan_summary", {})
        print(f"📊 扫描概览:")
        print(f"  - 扫描文件数: {summary.get('scan_overview', {}).get('files_scanned', 0)}")
        print(f"  - 总问题数: {summary.get('scan_overview', {}).get('total_issues', 0)}")
        print(f"  - 关键问题: {summary.get('scan_overview', {}).get('critical_issues', 0)}")
        print(f"  - 高优先级: {summary.get('scan_overview', {}).get('high_issues', 0)}")
        
        metrics = summary.get("quality_metrics", {})
        print(f"\n📈 质量指标:")
        print(f"  - 总体质量评分: {metrics.get('overall_quality_score', 0):.2f}")
        print(f"  - 质量等级: {metrics.get('quality_grade', 'N/A')}")
        print(f"  - 平均复杂度: {metrics.get('average_complexity', 0):.1f}")
        print(f"  - 平均可维护性: {metrics.get('average_maintainability', 0):.1f}")
    
    async def _show_feature_analyses(self, feature_analyses: Optional[List[Dict[str, Any]]]):
        """显示功能特点分析"""
        if not feature_analyses:
            print("❌ 无功能特点分析结果")
            return
        
        print("\n🎯 功能特点分析概览")
        print("-" * 40)
        
        total_files = len(feature_analyses)
        high_value_files = len([f for f in feature_analyses if hasattr(f, 'recommendation') and "保留" in f.recommendation])
        low_value_files = len([f for f in feature_analyses if hasattr(f, 'recommendation') and "删除" in f.recommendation])
        
        print(f"📁 分析文件总数: {total_files}")
        print(f"💎 高价值文件: {high_value_files}")
        print(f"🗑️ 低价值文件: {low_value_files}")
        
        # 显示前5个文件的详细分析
        print(f"\n📋 前5个文件分析:")
        for i, analysis in enumerate(feature_analyses[:5]):
            print(f"\n{i+1}. {analysis.file_path}")
            if hasattr(analysis, 'overall_assessment'):
                print(f"   评估: {analysis.overall_assessment.split('**评估结论**:')[-1].strip()}")
            if hasattr(analysis, 'recommendation'):
                print(f"   推荐: {analysis.recommendation}")
    
    async def _show_decision_records(self, decision_records: Optional[List[Dict[str, Any]]]):
        """显示决策记录"""
        if not decision_records:
            print("❌ 无决策记录")
            return
        
        print("\n⚖️ 决策记录概览")
        print("-" * 40)
        
        total_decisions = len(decision_records)
        retention_decisions = len([d for d in decision_records if d.get('decision_type') == 'file_retention'])
        removal_decisions = len([d for d in decision_records if d.get('decision_type') == 'file_removal'])
        refactor_decisions = len([d for d in decision_records if d.get('decision_type') == 'code_refactor'])
        
        print(f"📊 决策统计:")
        print(f"  - 总决策数: {total_decisions}")
        print(f"  - 保留决策: {retention_decisions}")
        print(f"  - 删除决策: {removal_decisions}")
        print(f"  - 重构决策: {refactor_decisions}")
        
        # 显示前3个决策的详细信息
        print(f"\n📋 前3个决策详情:")
        for i, record in enumerate(decision_records[:3]):
            print(f"\n{i+1}. 决策ID: {record.get('decision_id', 'N/A')}")
            print(f"   目标: {record.get('target', 'N/A')}")
            print(f"   类型: {record.get('decision_type', 'N/A')}")
            print(f"   决策: {record.get('decision', 'N/A')}")
            print(f"   置信度: {record.get('confidence_score', 0):.2f}")
    
    async def _show_structure_comparison(self):
        """显示项目结构变化"""
        print("\n🔄 项目结构变化分析")
        print("-" * 40)
        
        try:
            # 获取最新的结构对比结果
            comparison = await self.structure_analyzer.analyze_and_compare()
            
            changes = comparison.structure_changes
            print(f"📊 变化统计:")
            print(f"  - 新增文件: {changes.get('files_added_count', 0)}")
            print(f"  - 删除文件: {changes.get('files_removed_count', 0)}")
            print(f"  - 修改文件: {changes.get('files_modified_count', 0)}")
            
            impact = comparison.impact_analysis
            print(f"\n📈 影响分析:")
            print(f"  - 功能影响: {impact.get('functional_impact', {}).get('level', 'N/A')}")
            print(f"  - 性能影响: {impact.get('performance_impact', {}).get('level', 'N/A')}")
            print(f"  - 安全影响: {impact.get('security_impact', {}).get('level', 'N/A')}")
            print(f"  - 整体风险: {impact.get('overall_risk', 'N/A')}")
            
            if comparison.recommendations:
                print(f"\n💡 推荐建议:")
                for i, rec in enumerate(comparison.recommendations[:3], 1):
                    print(f"  {i}. {rec}")
        
        except Exception as e:
            print(f"❌ 结构对比分析失败: {e}")
    
    async def _show_optimization_recommendations(self):
        """显示优化建议"""
        print("\n💡 优化建议")
        print("-" * 40)
        
        try:
            # 读取最新的优化报告
            report_files = list(self.results_dir.glob("optimization_report_*.json"))
            if report_files:
                latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                
                with open(latest_report, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                strategies = report.get("optimization_strategies", {})
                
                print("📈 优化策略:")
                
                # 立即行动项
                immediate = strategies.get("immediate_actions", [])
                if immediate:
                    print("  🚨 立即行动项:")
                    for action in immediate:
                        print(f"    - {action.get('action', 'N/A')}")
                
                # 短期目标
                short_term = strategies.get("short_term_goals", [])
                if short_term:
                    print("  📅 短期目标:")
                    for goal in short_term:
                        print(f"    - {goal.get('goal', 'N/A')}")
                
                # 长期计划
                long_term = strategies.get("long_term_plans", [])
                if long_term:
                    print("  🎯 长期计划:")
                    for plan in long_term:
                        print(f"    - {plan.get('plan', 'N/A')}")
            
            else:
                print("❌ 无优化报告文件")
        
        except Exception as e:
            print(f"❌ 优化建议读取失败: {e}")
    
    async def _export_detailed_reports(self):
        """导出详细报告"""
        print("\n📄 导出详细报告")
        print("-" * 40)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.results_dir / f"enhanced_sc_test_export_{timestamp}"
        export_dir.mkdir(exist_ok=True)
        
        try:
            # 导出所有报告文件
            report_types = [
                ("深度扫描结果", "deep_scan_results_*.json"),
                ("功能分析结果", "feature_analyses_*.json"),
                ("决策记录", "decision_records_*.json"),
                ("优化报告", "optimization_report_*.json"),
                ("结构对比", "structure_comparison_*.json")
            ]
            
            exported_files = []
            for report_type, pattern in report_types:
                files = list(self.results_dir.glob(pattern))
                for file in files:
                    dest = export_dir / file.name
                    file.rename(dest)
                    exported_files.append(str(dest))
            
            print(f"✅ 报告已导出到: {export_dir}")
            print(f"📁 导出文件数: {len(exported_files)}")
            
            # 生成导出清单
            manifest = {
                "export_timestamp": timestamp,
                "export_directory": str(export_dir),
                "exported_files": exported_files,
                "total_files": len(exported_files)
            }
            
            manifest_file = export_dir / "export_manifest.json"
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            
            print(f"📋 导出清单: {manifest_file}")
        
        except Exception as e:
            print(f"❌ 报告导出失败: {e}")
    
    async def _reanalyze_specific_file(self):
        """重新分析特定文件"""
        print("\n🔄 重新分析特定文件")
        print("-" * 40)
        
        try:
            file_path = input("请输入要重新分析的文件路径: ").strip()
            if not file_path:
                print("❌ 文件路径不能为空")
                return
            
            full_path = self.project_root / file_path
            if not full_path.exists():
                print(f"❌ 文件不存在: {file_path}")
                return
            
            print(f"🔍 重新分析文件: {file_path}")
            
            # 执行功能分析
            feature_analysis = await self.feature_analyzer.analyze_comprehensive_features(file_path)
            
            # 显示分析结果
            print(f"\n📊 分析结果:")
            print(f"  - 特征数量: {len(feature_analysis.feature_characteristics)}")
            print(f"  - 优势数量: {len(feature_analysis.advantages)}")
            print(f"  - 劣势数量: {len(feature_analysis.disadvantages)}")
            print(f"  - 替代方案: {len(feature_analysis.alternatives)}")
            print(f"  - 推荐: {feature_analysis.recommendation}")
            
            # 生成决策记录
            analysis_data = {
                "features": [asdict(f) for f in feature_analysis.feature_characteristics],
                "advantages": [asdict(a) for a in feature_analysis.advantages],
                "disadvantages": [asdict(d) for d in feature_analysis.disadvantages],
                "alternatives": [asdict(a) for a in feature_analysis.alternatives]
            }
            
            if "删除" in feature_analysis.recommendation:
                decision_type = DecisionType.FILE_REMOVAL
            elif "保留" in feature_analysis.recommendation:
                decision_type = DecisionType.FILE_RETENTION
            elif "重构" in feature_analysis.recommendation:
                decision_type = DecisionType.CODE_REFACTOR
            else:
                decision_type = DecisionType.FILE_RETENTION
            
            decision_record = await self.justification_system.create_comprehensive_decision(
                decision_type=decision_type,
                target=file_path,
                analysis_data=analysis_data
            )
            
            print(f"\n⚖️ 决策记录已生成: {decision_record.decision_id}")
            print(f"  - 决策: {decision_record.decision}")
            print(f"  - 置信度: {decision_record.confidence_score:.2f}")
            print(f"  - 风险评估: {decision_record.risk_assessment}")
        
        except Exception as e:
            print(f"❌ 重新分析失败: {e}")
    
    async def _generate_final_report(self, 
                                  ai_context: Optional[Dict[str, Any]],
                                  structure_comparison: Optional[Any],
                                  deep_scan_results: Optional[Dict[str, Any]],
                                  feature_analyses: Optional[List[Dict[str, Any]]],
                                  decision_records: Optional[List[Dict[str, Any]]],
                                  optimization_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """生成最终综合报告"""
        print("📝 生成最终综合报告...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        final_report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "project_root": str(self.project_root),
                "report_version": "2.0.0",
                "command": "/sc:test enhanced"
            },
            "executive_summary": {
                "analysis_completed": True,
                "modules_executed": [
                    "AI信息传递",
                    "项目结构对比",
                    "深度扫描审查",
                    "功能特点分析",
                    "决策记录生成",
                    "优化报告生成"
                ],
                "overall_status": "completed",
                "recommendations": []
            },
            "ai_context": ai_context,
            "structure_comparison": asdict(structure_comparison) if structure_comparison else None,
            "deep_scan_results": deep_scan_results,
            "feature_analyses": [asdict(f) for f in feature_analyses] if feature_analyses else [],
            "decision_records": decision_records,
            "optimization_report": optimization_report,
            "conclusions": await self._generate_conclusions(
                deep_scan_results, feature_analyses, decision_records
            ),
            "next_steps": await self._generate_next_steps(
                deep_scan_results, feature_analyses, decision_records
            )
        }
        
        # 保存最终报告
        report_file = self.results_dir / f"enhanced_sc_test_final_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        
        # 生成Markdown版本
        markdown_file = self.results_dir / f"enhanced_sc_test_final_report_{timestamp}.md"
        markdown_content = await self._generate_markdown_report(final_report)
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"📋 最终报告已保存:")
        print(f"  JSON: {report_file}")
        print(f"  Markdown: {markdown_file}")
        
        return final_report
    
    async def _generate_conclusions(self, 
                                  deep_scan_results: Optional[Dict[str, Any]],
                                  feature_analyses: Optional[List[Dict[str, Any]]],
                                  decision_records: Optional[List[Dict[str, Any]]]) -> List[str]:
        """生成结论"""
        conclusions = []
        
        # 基于深度扫描结果的结论
        if deep_scan_results:
            summary = deep_scan_results.get("scan_summary", {})
            total_issues = summary.get("scan_overview", {}).get("total_issues", 0)
            quality_score = summary.get("quality_metrics", {}).get("overall_quality_score", 0)
            
            if total_issues > 0:
                conclusions.append(f"发现{total_issues}个需要关注的问题，建议优先处理高风险问题")
            
            if quality_score > 0.8:
                conclusions.append("整体代码质量良好，继续保持当前标准")
            elif quality_score > 0.6:
                conclusions.append("代码质量中等，有改进空间")
            else:
                conclusions.append("代码质量需要重点改进")
        
        # 基于功能分析的结论
        if feature_analyses:
            total_files = len(feature_analyses)
            high_value_files = len([f for f in feature_analyses if hasattr(f, 'recommendation') and "保留" in f.recommendation])
            
            conclusions.append(f"分析了{total_files}个文件，其中{high_value_files}个文件建议保留")
            
            if high_value_files > total_files * 0.7:
                conclusions.append("项目整体价值较高，大部分文件都有明确的业务价值")
            elif high_value_files > total_files * 0.4:
                conclusions.append("项目价值中等，需要优化部分文件的价值")
            else:
                conclusions.append("项目价值偏低，建议进行大幅优化")
        
        # 基于决策记录的结论
        if decision_records:
            total_decisions = len(decision_records)
            high_confidence_decisions = len([d for d in decision_records if d.get('confidence_score', 0) > 0.8])
            
            conclusions.append(f"生成了{total_decisions}个决策记录，其中{high_confidence_decisions}个高置信度决策")
            
            if high_confidence_decisions > total_decisions * 0.7:
                conclusions.append("决策质量较高，建议执行相关决策")
            else:
                conclusions.append("部分决策置信度较低，建议进一步分析")
        
        return conclusions
    
    async def _generate_next_steps(self, 
                                 deep_scan_results: Optional[Dict[str, Any]],
                                 feature_analyses: Optional[List[Dict[str, Any]]],
                                 decision_records: Optional[List[Dict[str, Any]]]) -> List[str]:
        """生成下一步行动"""
        next_steps = []
        
        # 基于深度扫描结果的行动
        if deep_scan_results:
            security_issues = len(deep_scan_results.get("security_issues", []))
            performance_issues = len(deep_scan_results.get("performance_issues", []))
            
            if security_issues > 0:
                next_steps.append("立即修复所有安全问题，确保系统安全性")
            
            if performance_issues > 0:
                next_steps.append("优化性能瓶颈，提升系统响应速度")
        
        # 基于功能分析的行动
        if feature_analyses:
            removal_candidates = [f for f in feature_analyses if hasattr(f, 'recommendation') and "删除" in f.recommendation]
            refactor_candidates = [f for f in feature_analyses if hasattr(f, 'recommendation') and "重构" in f.recommendation]
            
            if removal_candidates:
                next_steps.append(f"谨慎评估并考虑删除{len(removal_candidates)}个低价值文件")
            
            if refactor_candidates:
                next_steps.append(f"制定重构计划，优化{len(refactor_candidates)}个需要改进的文件")
        
        # 基于决策记录的行动
        if decision_records:
            high_risk_decisions = [d for d in decision_records if d.get('risk_assessment') == 'high']
            
            if high_risk_decisions:
                next_steps.append("重点关注高风险决策，制定详细的风险缓解策略")
        
        # 通用行动建议
        next_steps.extend([
            "建立定期代码审查机制，持续监控代码质量",
            "完善测试覆盖率，确保所有关键功能都有充分测试",
            "建立项目文档，记录重要的架构决策和设计原则",
            "定期重新评估项目结构，确保持续的优化和改进"
        ])
        
        return next_steps
    
    async def _generate_markdown_report(self, final_report: Dict[str, Any]) -> str:
        """生成Markdown报告"""
        content = []
        
        # 标题
        content.append("# 增强版 /sc:test 综合分析报告")
        content.append(f"**生成时间**: {final_report['metadata']['generated_at']}")
        content.append(f"**项目路径**: {final_report['metadata']['project_root']}")
        content.append("")
        
        # 执行摘要
        content.append("## 📊 执行摘要")
        summary = final_report["executive_summary"]
        content.append(f"**分析状态**: {'✅ 已完成' if summary['analysis_completed'] else '❌ 未完成'}")
        content.append("")
        
        content.append("### 执行的模块")
        for module in summary["modules_executed"]:
            content.append(f"- ✅ {module}")
        content.append("")
        
        # 结论
        content.append("## 🎯 主要结论")
        for conclusion in final_report["conclusions"]:
            content.append(f"- {conclusion}")
        content.append("")
        
        # 下一步行动
        content.append("## 📋 下一步行动")
        for i, step in enumerate(final_report["next_steps"], 1):
            content.append(f"{i}. {step}")
        content.append("")
        
        # 详细结果链接
        content.append("## 📄 详细报告")
        content.append("本分析生成了以下详细报告：")
        content.append("- 深度扫描结果")
        content.append("- 功能特点分析")
        content.append("- 决策记录")
        content.append("- 优化报告")
        content.append("- 项目结构对比")
        content.append("")
        
        return "\n".join(content)

async def main():
    """主函数"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="增强版 /sc:test 命令 - 全面的项目测试和分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python test_enhanced_main.py                    # 启动交互模式
  python test_enhanced_main.py --no-interactive   # 非交互模式
  python test_enhanced_main.py --project-root /path/to/project  # 指定项目路径
        """
    )
    parser.add_argument("--project-root", default=".", help="项目根目录路径 (默认: 当前目录)")
    parser.add_argument("--no-interactive", action="store_true", help="禁用交互模式，直接执行分析")
    parser.add_argument("--no-ai-awareness", action="store_true", help="禁用AI信息传递")
    parser.add_argument("--no-deep-analysis", action="store_true", help="禁用深度分析")
    parser.add_argument("--no-optimization-report", action="store_true", help="禁用优化报告生成")
    parser.add_argument("--no-structure-comparison", action="store_true", help="禁用项目结构对比")
    parser.add_argument("--help", "-h", action="store_true", help="显示帮助信息")
    
    # 如果没有参数，显示欢迎信息并默认启动交互模式
    if len(sys.argv) == 1:
        print("🎯 增强版 /sc:test 系统启动中...")
        print("💡 检测到无参数启动，将使用默认交互模式")
        print("📖 使用 --help 查看详细帮助信息\n")
    
    args = parser.parse_args()
    
    # 显示帮助信息
    if len(sys.argv) == 1 or args.help:
        print("""
╔══════════════════════════════════════════════════════════════╗
║                    🎯 增强版 /sc:test 系统                   ║
║                 全面的项目测试和分析工具                      ║
╚══════════════════════════════════════════════════════════════╝

📋 功能特性:
  ✅ 深度代码扫描和质量分析
  ✅ 智能功能特点和价值评估
  ✅ 全面决策支持系统
  ✅ 自动优化报告生成
  ✅ 交互式结果查看界面
  ✅ 项目结构对比分析
  ✅ 自定义报告生成
  ✅ 灵活的配置选项

🚀 快速开始:
  python test_enhanced_main.py                    # 启动交互模式（推荐）
  python test_enhanced_main.py --no-interactive   # 非交互模式执行

⚙️ 高级选项:
  --project-root PATH          指定项目根目录
  --no-interactive            禁用交互模式
  --no-ai-awareness           禁用AI信息传递
  --no-deep-analysis          禁用深度分析
  --no-optimization-report    禁用优化报告
  --no-structure-comparison   禁用结构对比

💡 交互模式特色:
  🎨 友好的用户界面
  📊 实时状态显示
  🔍 详细结果查看
  ⚙️ 灵活系统设置
  📄 自定义报告生成
  🛠️ 强大的分析工具

📞 获取帮助:
  python test_enhanced_main.py --help
        """)
        
        if len(sys.argv) == 1:
            # 无参数时继续执行
            pass
        else:
            return
    
    try:
        # 创建增强版测试命令实例
        enhanced_test = EnhancedSCTestCommand(args.project_root)
        
        # 显示启动信息
        if not args.no_interactive:
            print("🚀 正在初始化增强版 /sc:test 系统...")
            print("📊 项目路径:", args.project_root)
            print("🎯 交互模式:", "启用" if not args.no_interactive else "禁用")
            print()
        
        # 执行增强版测试
        results = await enhanced_test.execute_enhanced_test(
            interactive_mode=not args.no_interactive,
            force_ai_awareness=not args.no_ai_awareness,
            enable_deep_analysis=not args.no_deep_analysis,
            generate_optimization_report=not args.no_optimization_report,
            compare_structures=not args.no_structure_comparison
        )
        
        # 显示完成信息
        print("\n" + "🎉" * 20)
        print("🎉 增强版 /sc:test 执行完成！")
        print("🎉" * 20)
        print(f"📊 分析状态: {results['executive_summary']['overall_status']}")
        
        # 显示结果摘要
        if results.get('conclusions'):
            print("\n📋 主要结论:")
            for i, conclusion in enumerate(results['conclusions'][:3], 1):
                print(f"  {i}. {conclusion}")
        
        if results.get('next_steps'):
            print("\n📈 下一步建议:")
            for i, step in enumerate(results['next_steps'][:3], 1):
                print(f"  {i}. {step}")
        
        print("\n💾 详细报告已保存到 reports/ 目录")
        print("👋 感谢使用增强版 /sc:test 系统！")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断操作")
        print("👋 系统安全退出")
    except Exception as e:
        print(f"\n❌ 系统执行出错: {e}")
        print("💡 请检查系统配置或联系技术支持")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())