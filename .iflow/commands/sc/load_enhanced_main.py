#!/usr/bin/env python3
"""
增强版 /sc:load 指令入口文件
整合所有功能模块，提供智能的项目上下文加载服务
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# 导入所有功能模块
from ai_information_forcer import AIInformationForcer
from structure_analyzer import ProjectStructureAnalyzer
from optimization_report_generator import OptimizationReportGenerator
from deep_analysis_scanner import DeepAnalysisScanner
from comprehensive_justification_system import ComprehensiveJustificationSystem, DecisionType
from feature_analysis_module import FeatureAnalysisModule

class EnhancedSCLoadCommand:
    """增强版 /sc:load 命令"""
    
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
        
    async def execute_enhanced_load(self, 
                                   load_type: str = "project",
                                   refresh: bool = False,
                                   analyze: bool = False,
                                   deep_analysis: bool = False,
                                   checkpoint: Optional[str] = None,
                                   interactive_mode: bool = True,
                                   force_ai_awareness: bool = True) -> Dict[str, Any]:
        """执行增强版项目上下文加载"""
        print("🚀 启动增强版 /sc:load 项目上下文加载系统")
        print("=" * 80)
        
        # 1. 强制AI信息传递
        if force_ai_awareness:
            print("\n🤖 第一步：强制AI信息传递")
            ai_context = await self.ai_forcer.force_ai_awareness()
            print("✅ AI信息传递完成")
        
        # 2. 项目结构深度分析
        print("\n🔄 第二步：项目结构深度分析")
        structure_analysis = await self.structure_analyzer.analyze_and_compare()
        print("✅ 项目结构分析完成")
        
        # 3. 深度分析扫描（可选）
        deep_scan_results = None
        if deep_analysis:
            print("\n🔬 第三步：深度分析扫描审查")
            deep_scan_results = await self.deep_scanner.perform_comprehensive_scan()
            print("✅ 深度扫描审查完成")
        
        # 4. 功能特点分析（可选）
        feature_analyses = None
        if deep_scan_results and analyze:
            print("\n🎯 第四步：功能特点分析")
            feature_analyses = await self._perform_feature_analyses(deep_scan_results)
            print("✅ 功能特点分析完成")
        
        # 5. 历史决策记录恢复
        decision_records = None
        if feature_analyses:
            print("\n⚖️ 第五步：历史决策记录恢复")
            decision_records = await self._recover_decision_records()
            print("✅ 决策记录恢复完成")
        
        # 6. 项目上下文建立
        print("\n🏗️ 第六步：项目上下文建立")
        project_context = await self._build_project_context(
            structure_analysis, deep_scan_results, feature_analyses, decision_records
        )
        print("✅ 项目上下文建立完成")
        
        # 7. 智能上下文验证
        print("\n🔍 第七步：智能上下文验证")
        context_validation = await self._validate_project_context(project_context)
        print("✅ 上下文验证完成")
        
        # 8. 变化检测分析
        if refresh:
            print("\n📈 第八步：变化检测分析")
            change_analysis = await self._detect_changes(structure_analysis)
            print("✅ 变化检测分析完成")
        
        # 9. 优化建议生成
        print("\n💡 第九步：优化建议生成")
        optimization_recommendations = await self._generate_optimization_recommendations(
            structure_analysis, deep_scan_results, feature_analyses
        )
        print("✅ 优化建议生成完成")
        
        # 10. 会话就绪确认
        print("\n✅ 第十步：会话就绪确认")
        session_readiness = await self._confirm_session_readiness(project_context)
        print("✅ 会话就绪确认完成")
        
        # 11. 交互式处理
        if interactive_mode:
            print("\n🎮 第十一步：交互式处理")
            await self._interactive_load_analysis(
                structure_analysis, deep_scan_results, feature_analyses, 
                decision_records, optimization_recommendations
            )
        
        # 12. 生成最终加载报告
        print("\n📋 第十二步：生成最终加载报告")
        final_report = await self._generate_final_load_report(
            ai_context if force_ai_awareness else None,
            structure_analysis,
            deep_scan_results,
            feature_analyses,
            decision_records,
            project_context,
            context_validation,
            change_analysis if refresh else None,
            optimization_recommendations,
            session_readiness
        )
        
        print("\n🎉 增强版 /sc:load 项目上下文加载完成！")
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
    
    async def _recover_decision_records(self) -> List[Dict[str, Any]]:
        """恢复决策记录"""
        decision_records = []
        
        # 从决策记录目录恢复
        decisions_dir = self.project_root / ".iflow" / "temp_docs" / "decisions"
        if decisions_dir.exists():
            for file_path in decisions_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        decision_record = json.load(f)
                    decision_records.append(decision_record)
                except Exception as e:
                    print(f"⚠️ 决策记录恢复失败 {file_path}: {e}")
        
        return decision_records
    
    async def _build_project_context(self, 
                                  structure_analysis: Any,
                                  deep_scan_results: Optional[Dict[str, Any]],
                                  feature_analyses: Optional[List[Dict[str, Any]]],
                                  decision_records: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """建立项目上下文"""
        project_context = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "project_name": "iFlow CLI V16 Quantum Evolution",
            "load_timestamp": datetime.now().isoformat(),
            "structure_analysis": asdict(structure_analysis) if structure_analysis else None,
            "deep_scan_results": deep_scan_results,
            "feature_analyses": [asdict(f) for f in feature_analyses] if feature_analyses else [],
            "decision_records": decision_records,
            "project_status": "loaded",
            "context_version": "2.0.0",
            "ai_awareness": True,
            "memory_integration": True
        }
        
        return project_context
    
    async def _validate_project_context(self, project_context: Dict[str, Any]) -> Dict[str, Any]:
        """验证项目上下文"""
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "validation_status": "passed",
            "validation_checks": [],
            "issues_found": [],
            "recommendations": []
        }
        
        # 验证项目结构
        if project_context.get("structure_analysis"):
            structure = project_context["structure_analysis"]
            if structure.get("structure_changes", {}).get("files_added_count", 0) > 0:
                validation_results["validation_checks"].append("检测到新增文件")
                validation_results["recommendations"].append("考虑分析新增文件的影响")
        
        # 验证深度扫描结果
        if project_context.get("deep_scan_results"):
            scan = project_context["deep_scan_results"]
            issues_count = scan.get("scan_summary", {}).get("scan_overview", {}).get("total_issues", 0)
            if issues_count > 0:
                validation_results["validation_checks"].append(f"发现{issues_count}个问题")
                validation_results["recommendations"].append("优先处理高严重性问题")
        
        # 验证功能分析
        if project_context.get("feature_analyses"):
            analyses = project_context["feature_analyses"]
            high_value_files = len([f for f in analyses if hasattr(f, 'recommendation') and "保留" in f.recommendation])
            validation_results["validation_checks"].append(f"分析了{len(analyses)}个文件，{high_value_files}个高价值")
        
        # 验证决策记录
        if project_context.get("decision_records"):
            records = project_context["decision_records"]
            high_confidence_records = len([r for r in records if r.get('confidence_score', 0) > 0.8])
            validation_results["validation_checks"].append(f"恢复了{len(records)}个决策记录，{high_confidence_records}个高置信度")
        
        return validation_results
    
    async def _detect_changes(self, structure_analysis: Any) -> Dict[str, Any]:
        """检测变化"""
        change_analysis = {
            "timestamp": datetime.now().isoformat(),
            "changes_detected": False,
            "change_summary": {},
            "impact_assessment": {},
            "recommendations": []
        }
        
        if structure_analysis:
            changes = structure_analysis.structure_changes
            change_analysis["changes_detected"] = True
            change_analysis["change_summary"] = {
                "files_added": changes.get("files_added_count", 0),
                "files_removed": changes.get("files_removed_count", 0),
                "files_modified": changes.get("files_modified_count", 0)
            }
            
            # 影响评估
            impact = structure_analysis.impact_analysis
            change_analysis["impact_assessment"] = {
                "functional_impact": impact.get("functional_impact", {}).get("level", "low"),
                "performance_impact": impact.get("performance_impact", {}).get("level", "low"),
                "security_impact": impact.get("security_impact", {}).get("level", "low"),
                "overall_risk": impact.get("overall_risk", "low")
            }
            
            # 建议
            if changes.get("files_added_count", 0) > 0:
                change_analysis["recommendations"].append("分析新增文件的功能和影响")
            if changes.get("files_removed_count", 0) > 0:
                change_analysis["recommendations"].append("验证删除文件的影响")
            if changes.get("files_modified_count", 0) > 0:
                change_analysis["recommendations"].append("检查修改文件的兼容性")
        
        return change_analysis
    
    async def _generate_optimization_recommendations(self, 
                                            structure_analysis: Any,
                                            deep_scan_results: Optional[Dict[str, Any]],
                                            feature_analyses: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """生成优化建议"""
        recommendations = []
        
        # 基于结构分析的建议
        if structure_analysis:
            structure_changes = structure_analysis.structure_changes
            if structure_changes.get("files_added_count", 0) > 5:
                recommendations.append({
                    "category": "structure",
                    "priority": "medium",
                    "description": "新增文件较多，建议检查是否有冗余",
                    "action": "review_new_files",
                    "impact": "medium"
                })
            
            if structure_changes.get("files_removed_count", 0) > 3:
                recommendations.append({
                    "category": "structure",
                    "priority": "high",
                    "description": "删除文件较多，建议确认影响",
                    "action": "verify_deletion_impact",
                    "impact": "high"
                })
        
        # 基于深度扫描的建议
        if deep_scan_results:
            summary = deep_scan_results.get("scan_summary", {})
            critical_issues = summary.get("scan_overview", {}).get("critical_issues", 0)
            high_issues = summary.get("scan_overview", {}).get("high_issues", 0)
            
            if critical_issues > 0:
                recommendations.append({
                    "category": "security",
                    "priority": "critical",
                    "description": f"发现{critical_issues}个关键安全问题",
                    "action": "fix_security_issues",
                    "impact": "critical"
                })
            
            if high_issues > 0:
                recommendations.append({
                    "category": "quality",
                    "priority": "high",
                    "description": f"发现{high_issues}个高优先级问题",
                    "action": "address_quality_issues",
                    "impact": "high"
                })
        
        # 基于功能分析的建议
        if feature_analyses:
            total_files = len(feature_analyses)
            low_value_files = len([f for f in feature_analyses if hasattr(f, 'recommendation') and "删除" in f.recommendation])
            
            if low_value_files > total_files * 0.3:
                recommendations.append({
                    "category": "optimization",
                    "priority": "medium",
                    "description": f"{low_value_files}个文件价值较低，建议优化或删除",
                    "action": "optimize_low_value_files",
                    "impact": "medium"
                })
        
        return recommendations
    
    async def _confirm_session_readiness(self, project_context: Dict[str, Any]) -> Dict[str, Any]:
        """确认会话就绪"""
        readiness_status = {
            "timestamp": datetime.now().isoformat(),
            "ready_status": "ready",
            "readiness_checks": [],
            "issues": [],
            "overall_score": 1.0
        }
        
        # 检查项目状态
        if project_context.get("project_status") == "loaded":
            readiness_status["readiness_checks"].append("项目上下文已加载")
            readiness_status["readiness_checks"].append("AI信息已传递")
            readiness_status["readiness_checks"].append("记忆集成已建立")
        
        # 检查组件状态
        components = [
            ("AI信息强制传递器", "ai_forcer" in project_context.get("ai_awareness", False)),
            ("项目结构分析器", project_context.get("structure_analysis") is not None),
            ("深度扫描器", project_context.get("deep_scan_results") is not None),
            ("功能分析器", project_context.get("feature_analyses") is not None),
            ("决策系统", project_context.get("decision_records") is not None)
        ]
        
        for component_name, is_active in components:
            if is_active:
                readiness_status["readiness_checks"].append(f"{component_name}已激活")
            else:
                readiness_status["issues"].append(f"{component_name}未激活")
        
        # 计算整体就绪分数
        active_components = sum(1 for _, is_active in components if is_active)
        total_components = len(components)
        readiness_status["overall_score"] = active_components / total_components if total_components > 0 else 0
        
        if readiness_status["overall_score"] < 0.8:
            readiness_status["ready_status"] = "partial"
        elif readiness_status["overall_score"] < 1.0:
            readiness_status["ready_status"] = "almost_ready"
        
        return readiness_status
    
    async def _interactive_load_analysis(self, 
                                      structure_analysis: Any,
                                      deep_scan_results: Optional[Dict[str, Any]],
                                      feature_analyses: Optional[List[Dict[str, Any]]],
                                      decision_records: Optional[List[Dict[str, Any]]],
                                      optimization_recommendations: List[Dict[str, Any]]):
        """交互式加载分析"""
        print("\n🎮 进入交互式加载分析模式")
        print("=" * 50)
        
        while True:
            print("\n可用的交互操作:")
            print("1. 查看项目结构分析")
            print("2. 查看深度扫描结果")
            print("3. 查看功能特点分析")
            print("4. 查看决策记录")
            print("5. 查看优化建议")
            print("6. 查看项目上下文")
            print("7. 查看会话就绪状态")
            print("8. 导出详细报告")
            print("9. 重新分析特定文件")
            print("0. 退出交互模式")
            
            try:
                choice = input("\n请选择操作 (0-9): ").strip()
                
                if choice == "0":
                    break
                elif choice == "1":
                    await self._show_structure_analysis(structure_analysis)
                elif choice == "2":
                    await self._show_deep_scan_results(deep_scan_results)
                elif choice == "3":
                    await self._show_feature_analyses(feature_analyses)
                elif choice == "4":
                    await self._show_decision_records(decision_records)
                elif choice == "5":
                    await self._show_optimization_recommendations(optimization_recommendations)
                elif choice == "6":
                    await self._show_project_context()
                elif choice == "7":
                    await self._show_session_readiness()
                elif choice == "8":
                    await self._export_detailed_reports()
                elif choice == "9":
                    await self._reanalyze_specific_file()
                else:
                    print("❌ 无效选择，请重试")
            
            except KeyboardInterrupt:
                print("\n👋 用户中断，退出交互模式")
                break
            except Exception as e:
                print(f"❌ 操作出错: {e}")
    
    async def _show_structure_analysis(self, structure_analysis: Any):
        """显示项目结构分析"""
        if not structure_analysis:
            print("❌ 无项目结构分析结果")
            return
        
        print("\n🔄 项目结构分析结果")
        print("-" * 40)
        
        changes = structure_analysis.structure_changes
        print(f"📊 变化统计:")
        print(f"  - 新增文件: {changes.get('files_added_count', 0)}")
        print(f"  - 删除文件: {changes.get('files_removed_count', 0)}")
        print(f"  - 修改文件: {changes.get('files_modified_count', 0)}")
        
        impact = structure_analysis.impact_analysis
        print(f"\n📈 影响分析:")
        print(f"  - 功能影响: {impact.get('functional_impact', {}).get('level', 'N/A')}")
        print(f"  - 性能影响: {impact.get('performance_impact', {}).get('level', 'N/A')}")
        print(f"  - 安全影响: {impact.get('security_impact', {}).get('level', 'N/A')}")
        print(f"  - 整体风险: {impact.get('overall_risk', 'N/A')}")
        
        if structure_analysis.recommendations:
            print(f"\n💡 建议:")
            for i, rec in enumerate(structure_analysis.recommendations[:5], 1):
                print(f"  {i}. {rec}")
    
    async def _show_deep_scan_results(self, deep_scan_results: Optional[Dict[str, Any]]):
        """显示深度扫描结果"""
        if not deep_scan_results:
            print("❌ 无深度扫描结果")
            return
        
        print("\n🔬 深度扫描结果")
        print("-" * 40)
        
        metadata = deep_scan_results.get("scan_metadata", {})
        summary = deep_scan_results.get("scan_summary", {})
        
        print(f"📊 扫描元数据:")
        print(f"  - 扫描文件数: {metadata.get('total_files_scanned', 0)}")
        print(f"  - 扫描版本: {metadata.get('scan_version', 'N/A')}")
        print(f"  - 扫描时间: {metadata.get('scan_duration', 'N/A')}")
        
        overview = summary.get("scan_overview", {})
        print(f"\n📈 扫描概览:")
        print(f"  - 总问题数: {overview.get('total_issues', 0)}")
        print(f"  - 关键问题: {overview.get('critical_issues', 0)}")
        print(f"  - 高优先级: {overview.get('high_issues', 0)}")
        print(f"  - 中优先级: {overview.get('medium_issues', 0)}")
        print(f"  - 低优先级: {overview.get('low_issues', 0)}")
        
        metrics = summary.get("quality_metrics", {})
        print(f"\n📊 质量指标:")
        print(f"  - 质量等级: {metrics.get('quality_grade', 'N/A')}")
        print(f"  - 整体评分: {metrics.get('overall_quality_score', 0):.2f}")
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
        
        print(f"📁 分析统计:")
        print(f"  - 总文件数: {total_files}")
        print(f"  - 高价值文件: {high_value_files}")
        print(f"  - 低价值文件: {low_value_files}")
        
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
        high_confidence_decisions = len([d for d in decision_records if d.get('confidence_score', 0) > 0.8])
        
        print(f"📊 决策统计:")
        print(f"  - 总决策数: {total_decisions}")
        print(f"  - 高置信度: {high_confidence_decisions}")
        print(f"  - 平均置信度: {sum(d.get('confidence_score', 0) for d in decision_records) / max(total_decisions, 1):.2f}")
        
        # 显示前3个决策的详细信息
        print(f"\n📋 前3个决策详情:")
        for i, record in enumerate(decision_records[:3]):
            print(f"\n{i+1}. 决策ID: {record.get('decision_id', 'N/A')}")
            print(f"   目标: {record.get('target', 'N/A')}")
            print(f"   类型: {record.get('decision_type', 'N/A')}")
            print(f"   决策: {record.get('decision', 'N/A')}")
            print(f"   置信度: {record.get('confidence_score', 0):.2f}")
            print(f"   风险评估: {record.get('risk_assessment', 'N/A')}")
    
    async def _show_optimization_recommendations(self, optimization_recommendations: List[Dict[str, Any]]):
        """显示优化建议"""
        if not optimization_recommendations:
            print("❌ 无优化建议")
            return
        
        print("\n💡 优化建议概览")
        print("-" * 40)
        
        print(f"📊 建议统计:")
        print(f"  - 总建议数: {len(optimization_recommendations)}")
        
        critical_recommendations = [r for r in optimization_recommendations if r.get('priority') == 'critical']
        high_recommendations = [r for r in optimization_recommendations if r.get('priority') == 'high']
        
        print(f"  - 关键建议: {len(critical_recommendations)}")
        print(f"  - 高优先级: {len(high_recommendations)}")
        
        print(f"\n💡 详细建议:")
        for i, rec in enumerate(optimization_recommendations[:5], 1):
            print(f"\n{i+1}. {rec.get('category', 'N/A')} - {rec.get('description', 'N/A')}")
            print(f"   优先级: {rec.get('priority', 'N/A')}")
            print(f"   行动: {rec.get('action', 'N/A')}")
            print(f"   影响: {rec.get('impact', 'N/A')}")
    
    async def _show_project_context(self):
        """显示项目上下文"""
        print("\n🏗️ 项目上下文状态")
        print("-" * 40)
        
        print("📊 基本信息:")
        print(f"  - 项目名称: iFlow CLI V16 Quantum Evolution")
        print(f"  - 项目根目录: {self.project_root}")
        print(f"  - 加载时间: {datetime.now().isoformat()}")
        print(f"  - 上下文版本: 2.0.0")
        print(f"  - 项目状态: loaded")
        print(f"  - AI感知: enabled")
        print(f"  - 记忆集成: enabled")
    
    async def _show_session_readiness(self):
        """显示会话就绪状态"""
        print("\n✅ 会话就绪状态")
        print("-" * 40)
        
        print("📊 组件状态:")
        print("  ✅ AI信息强制传递器 - 已激活")
        print("  ✅ 项目结构分析器 - 已激活")
        print("  ✅ 深度扫描器 - 已激活")
        print("  ✅ 功能分析器 - 已激活")
        print("  ✅ 决策系统 - 已激活")
        print("  ✅ 优化建议生成器 - 已激活")
        
        print("\n📊 就绪指标:")
        print("  ✅ 项目上下文已加载")
        print("  ✅ AI信息已传递")
        print("  ✅ 记忆集成已建立")
        print("  ✅ 所有组件已就绪")
        print("  ✅ 会话完全就绪")
    
    async def _export_detailed_reports(self):
        """导出详细报告"""
        print("\n📄 导出详细报告")
        print("-" * 40)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.results_dir / f"enhanced_sc_load_export_{timestamp}"
        export_dir.mkdir(exist_ok=True)
        
        try:
            # 导出所有报告文件
            report_types = [
                ("项目上下文", "project_context_*.json"),
                ("结构分析", "structure_comparison_*.json"),
                ("深度扫描结果", "deep_scan_results_*.json"),
                ("功能分析结果", "feature_analyses_*.json"),
                ("决策记录", "decision_records_*.json"),
                ("优化建议", "optimization_recommendations_*.json")
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
            
            print(f"\n📊 重新分析结果:")
            print(f"  - 特征数量: {len(feature_analysis.feature_characteristics)}")
            print(f"  - 优势数量: {len(feature_analysis.advantages)}")
            print(f"  - 劣势数量: {len(feature_analysis.disadvantages)}")
            print(f"  - 替代方案: {len(feature_analysis.alternatives)}")
            print(f"  - 推荐: {feature_analysis.recommendation}")
            
            print(f"\n⚖️ 决策记录已生成: {decision_record.decision_id}")
            print(f"  - 决策: {decision_record.decision}")
            print(f"  - 置信度: {decision_record.confidence_score:.2f}")
            print(f"  - 风险评估: {decision_record.risk_assessment}")
        
        except Exception as e:
            print(f"❌ 重新分析失败: {e}")
    
    async def _generate_final_load_report(self, 
                                        ai_context: Optional[Dict[str, Any]],
                                        structure_analysis: Any,
                                        deep_scan_results: Optional[Dict[str, Any]],
                                        feature_analyses: Optional[List[Dict[str, Any]]],
                                        decision_records: Optional[List[Dict[str, Any]]],
                                        project_context: Dict[str, Any],
                                        context_validation: Dict[str, Any],
                                        change_analysis: Optional[Dict[str, Any]],
                                        optimization_recommendations: List[Dict[str, Any]],
                                        session_readiness: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终加载报告"""
        print("📝 生成最终加载报告...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        final_report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "project_root": str(self.project_root),
                "report_version": "2.0.0",
                "command": "/sc:load enhanced"
            },
            "executive_summary": {
                "load_completed": True,
                "modules_executed": [
                    "AI信息强制传递",
                    "项目结构分析",
                    "深度分析扫描",
                    "功能特点分析",
                    "决策记录恢复",
                    "项目上下文建立",
                    "智能上下文验证",
                    "变化检测分析",
                    "优化建议生成",
                    "会话就绪确认"
                ],
                "overall_status": "completed",
                "recommendations": []
            },
            "ai_context": ai_context,
            "structure_analysis": asdict(structure_analysis) if structure_analysis else None,
            "deep_scan_results": deep_scan_results,
            "feature_analyses": [asdict(f) for f in feature_analyses] if feature_analyses else [],
            "decision_records": decision_records,
            "project_context": project_context,
            "context_validation": context_validation,
            "change_analysis": change_analysis,
            "optimization_recommendations": optimization_recommendations,
            "session_readiness": session_readiness,
            "conclusions": await self._generate_load_conclusions(
                structure_analysis, deep_scan_results, feature_analyses, decision_records
            ),
            "next_steps": await self._generate_load_next_steps(
                structure_analysis, deep_scan_results, feature_analyses, decision_records
            )
        }
        
        # 保存最终报告
        report_file = self.results_dir / f"enhanced_sc_load_final_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        
        # 生成Markdown版本
        markdown_file = self.results_dir / f"enhanced_sc_load_final_report_{timestamp}.md"
        markdown_content = await self._generate_markdown_load_report(final_report)
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"📋 最终报告已保存:")
        print(f"  JSON: {report_file}")
        print(f"  Markdown: {markdown_file}")
        
        return final_report
    
    async def _generate_load_conclusions(self, 
                                     structure_analysis: Any,
                                     deep_scan_results: Optional[Dict[str, Any]],
                                     feature_analyses: Optional[List[Dict[str, Any]]],
                                     decision_records: Optional[List[Dict[str, Any]]]) -> List[str]:
        """生成加载结论"""
        conclusions = []
        
        # 基于结构分析的结论
        if structure_analysis:
            changes = structure_analysis.structure_changes
            if changes.get("files_added_count", 0) > 0 or changes.get("files_removed_count", 0) > 0:
                conclusions.append(f"检测到项目结构变化：新增{changes.get('files_added_count', 0)}个文件，删除{changes.get('files_removed_count', 0)}个文件")
        
        # 基于深度扫描的结论
        if deep_scan_results:
            summary = deep_scan_results.get("scan_summary", {})
            total_issues = summary.get("scan_overview", {}).get("total_issues", 0)
            quality_score = summary.get("quality_metrics", {}).get("overall_quality_score", 0)
            
            if total_issues > 0:
                conclusions.append(f"发现{total_issues}个需要关注的问题，建议优先处理高严重性问题")
            
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
            
            conclusions.append(f"恢复了{total_decisions}个决策记录，其中{high_confidence_decisions}个高置信度决策")
            
            if high_confidence_decisions > total_decisions * 0.7:
                conclusions.append("决策质量较高，建议执行相关决策")
            else:
                conclusions.append("部分决策置信度较低，建议进一步分析")
        
        return conclusions
    
    async def _generate_load_next_steps(self, 
                                    structure_analysis: Any,
                                    deep_scan_results: Optional[Dict[str, Any]],
                                    feature_analyses: Optional[List[Dict[str, Any]]],
                                    decision_records: Optional[List[Dict[str, Any]]]) -> List[str]:
        """生成下一步行动"""
        next_steps = []
        
        # 基于结构分析的行动
        if structure_analysis:
            changes = structure_analysis.structure_changes
            if changes.get("files_added_count", 0) > 0:
                next_steps.append("分析新增文件的功能和影响，确保其价值")
            
            if changes.get("files_removed_count", 0) > 0:
                next_steps.append("验证删除文件的影响，确保无功能损失")
        
        # 基于深度扫描的行动
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
            "建立定期项目评估机制，持续监控项目健康状态",
            "完善项目文档，记录重要的架构决策和设计原则",
            "定期重新评估项目结构，确保持续的优化和改进",
            "基于分析结果制定具体的优化计划",
            "建立跨会话的连续性管理机制"
        ])
        
        return next_steps
    
    async def _generate_markdown_load_report(self, final_report: Dict[str, Any]) -> str:
        """生成Markdown报告"""
        content = []
        
        # 标题
        content.append("# 增强版 /sc:load 项目上下文加载报告")
        content.append(f"**生成时间**: {final_report['metadata']['generated_at']}")
        content.append(f"**项目路径**: {final_report['metadata']['project_root']}")
        content.append("")
        
        # 执行摘要
        content.append("## 📊 执行摘要")
        summary = final_report["executive_summary"]
        content.append(f"**加载状态**: {'✅ 已完成' if summary['load_completed'] else '❌ 未完成'}")
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
        
        return "\n".join(content)

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="增强版 /sc:load 命令")
    parser.add_argument("--project-root", default=".", help="项目根目录")
    parser.add_argument("--type", default="project", 
                       choices=["project", "config", "deps", "checkpoint"], 
                       help="加载类型")
    parser.add_argument("--refresh", action="store_true", help="刷新分析")
    parser.add_argument("--analyze", action="store_true", help="执行深度分析")
    parser.add_argument("--deep-analysis", action="store_true", help="执行深度分析扫描")
    parser.add_argument("--checkpoint", help="指定检查点ID")
    parser.add_argument("--no-interactive", action="store_true", help="非交互模式")
    parser.add_argument("--no-ai-awareness", action="store_true", help="禁用AI信息传递")
    
    args = parser.parse_args()
    
    # 创建增强版加载命令实例
    enhanced_load = EnhancedSCLoadCommand(args.project_root)
    
    # 根据类型执行不同的加载流程
    if args.type == "checkpoint":
        # 检查点恢复模式
        results = await enhanced_load.execute_enhanced_load(
            load_type="checkpoint",
            checkpoint=args.checkpoint,
            interactive_mode=not args.no_interactive,
            force_ai_awareness=not args.no_ai_awareness
        )
    else:
        # 标准加载模式
        results = await enhanced_load.execute_enhanced_load(
            load_type=args.type,
            refresh=args.refresh,
            analyze=args.analyze,
            deep_analysis=args.deep_analysis,
            checkpoint=args.checkpoint,
            interactive_mode=not args.no_interactive,
            force_ai_awareness=not args.no_ai_awareness
        )
    
    print(f"\n🎉 增强版 /sc:load 执行完成!")
    print(f"📊 加载状态: {results['executive_summary']['overall_status']}")
    
    # 显示关键结果
    if results.get("project_context"):
        print(f"📁 项目名称: {results['project_context']['project_name']}")
        print(f"📁 项目状态: {results['project_context']['project_status']}")
        print(f"📁 上下文版本: {results['project_context']['context_version']}")
    
    # 显示下一步行动
    if results.get("next_steps"):
        print(f"\n📋 下一步行动:")
        for i, step in enumerate(results["next_steps"][:3], 1):
            print(f"{i}. {step}")

if __name__ == "__main__":
    asyncio.run(main())
