#!/usr/bin/env python3
"""
自动优化报告生成系统
基于项目结构树逐一排查，生成全面的优化报告
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import subprocess

@dataclass
class OptimizationItem:
    """优化项"""
    file_path: str
    issue_type: str
    severity: str
    description: str
    evidence: List[str]
    recommendation: str
    impact_assessment: str
    implementation_effort: str
    priority_score: float
    dependencies: List[str]

@dataclass
class FileAnalysisResult:
    """文件分析结果"""
    file_path: str
    file_size: int
    line_count: int
    function_count: int
    class_count: int
    import_count: int
    complexity_score: float
    maintainability_index: float
    duplication_ratio: float
    test_coverage: float
    security_issues: List[str]
    performance_issues: List[str]
    code_quality_issues: List[str]
    optimization_potential: float
    functionality_description: str
    advantages: List[str]
    disadvantages: List[str]
    retention_justification: str
    removal_justification: Optional[str]

class OptimizationReportGenerator:
    """优化报告生成器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合优化报告"""
        print("📈 开始生成综合优化报告...")
        
        # 1. 项目结构分析
        print("🔍 分析项目结构...")
        structure_analysis = await self._analyze_project_structure()
        
        # 2. 逐一文件分析
        print("📁 逐一分析文件...")
        file_analyses = await self._analyze_all_files(structure_analysis)
        
        # 3. 问题分类和优先级排序
        print("🏷️ 分类问题和排序优先级...")
        optimization_items = await self._classify_and_prioritize_issues(file_analyses)
        
        # 4. 生成优化策略
        print("💡 生成优化策略...")
        optimization_strategies = await self._generate_optimization_strategies(optimization_items)
        
        # 5. 影响评估
        print("📊 评估优化影响...")
        impact_assessment = await self._assess_optimization_impact(optimization_items)
        
        # 6. 实施计划
        print("📋 制定实施计划...")
        implementation_plan = await self._create_implementation_plan(optimization_items)
        
        # 7. 生成最终报告
        print("📝 生成最终报告...")
        final_report = await self._create_final_report(
            structure_analysis, file_analyses, optimization_items,
            optimization_strategies, impact_assessment, implementation_plan
        )
        
        # 8. 保存报告
        await self._save_report(final_report)
        
        print("✅ 综合优化报告生成完成")
        return final_report
    
    async def _analyze_project_structure(self) -> Dict[str, Any]:
        """分析项目结构"""
        structure = {
            "total_files": 0,
            "python_files": 0,
            "test_files": 0,
            "config_files": 0,
            "doc_files": 0,
            "directories": [],
            "file_tree": {},
            "size_distribution": {},
            "complexity_distribution": {}
        }
        
        file_sizes = []
        complexity_scores = []
        
        for root, dirs, files in os.walk(self.project_root):
            # 跳过隐藏目录和缓存
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            rel_root = os.path.relpath(root, self.project_root)
            if rel_root == '.':
                rel_root = 'root'
            
            structure["directories"].append(rel_root)
            structure["file_tree"][rel_root] = files
            
            for file in files:
                if not file.startswith('.'):
                    file_path = Path(root) / file
                    structure["total_files"] += 1
                    
                    if file.endswith('.py'):
                        structure["python_files"] += 1
                    elif 'test' in file.lower():
                        structure["test_files"] += 1
                    elif file in ['pyproject.toml', 'setup.cfg', 'requirements.txt']:
                        structure["config_files"] += 1
                    elif file.endswith('.md'):
                        structure["doc_files"] += 1
                    
                    # 收集文件大小
                    if file_path.exists():
                        size = file_path.stat().st_size
                        file_sizes.append(size)
                        
                        # 简单复杂度评估
                        if file.endswith('.py'):
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                complexity = len(content.split('\n')) + content.count('def ') * 2 + content.count('class ') * 3
                                complexity_scores.append(complexity)
                            except:
                                pass
        
        # 计算分布
        if file_sizes:
            structure["size_distribution"] = {
                "min": min(file_sizes),
                "max": max(file_sizes),
                "avg": sum(file_sizes) / len(file_sizes),
                "median": sorted(file_sizes)[len(file_sizes)//2]
            }
        
        if complexity_scores:
            structure["complexity_distribution"] = {
                "min": min(complexity_scores),
                "max": max(complexity_scores),
                "avg": sum(complexity_scores) / len(complexity_scores),
                "median": sorted(complexity_scores)[len(complexity_scores)//2]
            }
        
        return structure
    
    async def _analyze_all_files(self, structure_analysis: Dict[str, Any]) -> List[FileAnalysisResult]:
        """逐一分析所有文件"""
        file_analyses = []
        
        for root, dirs, files in os.walk(self.project_root):
            # 跳过隐藏目录和缓存
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    file_path = Path(root) / file
                    rel_path = str(file_path.relative_to(self.project_root))
                    
                    try:
                        analysis = await self._analyze_single_file(file_path)
                        file_analyses.append(analysis)
                    except Exception as e:
                        print(f"⚠️ 分析文件失败 {rel_path}: {e}")
        
        return file_analyses
    
    async def _analyze_single_file(self, file_path: Path) -> FileAnalysisResult:
        """分析单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # 基本统计
            line_count = len(lines)
            function_count = content.count('def ')
            class_count = content.count('class ')
            import_count = content.count('import')
            file_size = file_path.stat().st_size
            
            # 复杂度分析
            complexity_score = await self._calculate_complexity(content)
            maintainability_index = await self._calculate_maintainability_index(content)
            
            # 重复代码分析
            duplication_ratio = await self._analyze_duplication(content)
            
            # 测试覆盖率（简化）
            test_coverage = await self._estimate_test_coverage(file_path)
            
            # 问题检测
            security_issues = await self._detect_security_issues(content)
            performance_issues = await self._detect_performance_issues(content)
            code_quality_issues = await self._detect_code_quality_issues(content)
            
            # 优化潜力
            optimization_potential = await self._calculate_optimization_potential(
                security_issues, performance_issues, code_quality_issues
            )
            
            # 功能分析
            functionality_description = await self._analyze_functionality(content, file_path.name)
            advantages, disadvantages = await self._analyze_advantages_disadvantages(content, file_path.name)
            
            # 保留/删除理由
            retention_justification = await self._generate_retention_justification(
                functionality_description, advantages, disadvantages
            )
            removal_justification = await self._generate_removal_justification(
                functionality_description, disadvantages, security_issues
            )
            
            return FileAnalysisResult(
                file_path=str(file_path.relative_to(self.project_root)),
                file_size=file_size,
                line_count=line_count,
                function_count=function_count,
                class_count=class_count,
                import_count=import_count,
                complexity_score=complexity_score,
                maintainability_index=maintainability_index,
                duplication_ratio=duplication_ratio,
                test_coverage=test_coverage,
                security_issues=security_issues,
                performance_issues=performance_issues,
                code_quality_issues=code_quality_issues,
                optimization_potential=optimization_potential,
                functionality_description=functionality_description,
                advantages=advantages,
                disadvantages=disadvantages,
                retention_justification=retention_justification,
                removal_justification=removal_justification
            )
            
        except Exception as e:
            print(f"⚠️ 文件分析错误 {file_path}: {e}")
            # 返回默认分析结果
            return FileAnalysisResult(
                file_path=str(file_path.relative_to(self.project_root)),
                file_size=0,
                line_count=0,
                function_count=0,
                class_count=0,
                import_count=0,
                complexity_score=0.0,
                maintainability_index=0.0,
                duplication_ratio=0.0,
                test_coverage=0.0,
                security_issues=[f"分析错误: {e}"],
                performance_issues=[],
                code_quality_issues=[],
                optimization_potential=0.0,
                functionality_description="分析失败",
                advantages=[],
                disadvantages=[f"无法分析文件: {e}"],
                retention_justification="需要手动审查",
                removal_justification=None
            )
    
    async def _calculate_complexity(self, content: str) -> float:
        """计算复杂度"""
        complexity = 1.0  # 基础复杂度
        
        # 基于代码结构
        complexity += content.count('if ') * 0.5
        complexity += content.count('for ') * 0.5
        complexity += content.count('while ') * 0.5
        complexity += content.count('def ') * 0.3
        complexity += content.count('class ') * 0.5
        complexity += content.count('try:') * 0.3
        complexity += content.count('except ') * 0.3
        
        # 基于嵌套
        max_indent = 0
        for line in content.split('\n'):
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        
        complexity += max_indent * 0.1
        
        return complexity
    
    async def _calculate_maintainability_index(self, content: str) -> float:
        """计算可维护性指数"""
        lines = len([line for line in content.split('\n') if line.strip()])
        
        # 简化的可维护性指数计算
        base_score = 100.0
        
        # 代码量影响
        if lines > 1000:
            base_score -= 20
        elif lines > 500:
            base_score -= 10
        elif lines > 200:
            base_score -= 5
        
        # 复杂度影响
        complexity = await self._calculate_complexity(content)
        base_score -= complexity * 2
        
        # 注释影响（正面）
        comment_lines = content.count('#')
        comment_ratio = comment_lines / max(lines, 1)
        base_score += comment_ratio * 10
        
        return max(0, min(100, base_score))
    
    async def _analyze_duplication(self, content: str) -> float:
        """分析代码重复率"""
        lines = [line.strip() for line in content.split('\n') if line.strip() and len(line.strip()) > 10]
        
        if len(lines) < 10:
            return 0.0
        
        # 简单的重复检测
        unique_lines = set(lines)
        duplication_ratio = 1.0 - (len(unique_lines) / len(lines))
        
        return duplication_ratio
    
    async def _estimate_test_coverage(self, file_path: Path) -> float:
        """估算测试覆盖率"""
        # 简化的测试覆盖率估算
        # 检查是否有对应的测试文件
        test_patterns = [
            f"test_{file_path.stem}.py",
            f"{file_path.stem}_test.py",
            f"tests/test_{file_path.stem}.py"
        ]
        
        for pattern in test_patterns:
            test_file = file_path.parent / pattern
            if test_file.exists():
                return 0.8  # 假设有测试文件就有80%覆盖率
        
        # 检查文件名是否包含test
        if 'test' in file_path.name.lower():
            return 0.9  # 测试文件本身覆盖率很高
        
        return 0.3  # 默认覆盖率较低
    
    async def _detect_security_issues(self, content: str) -> List[str]:
        """检测安全问题"""
        issues = []
        
        # 危险函数
        dangerous_functions = ['eval(', 'exec(', 'compile(']
        for func in dangerous_functions:
            if func in content:
                issues.append(f"使用了危险函数: {func}")
        
        # 硬编码密码
        if re.search(r'password\s*=\s*["\'][^"\']+["\']', content, re.IGNORECASE):
            issues.append("可能存在硬编码密码")
        
        # SQL注入风险
        if 'execute(' in content and '%' in content:
            issues.append("可能存在SQL注入风险")
        
        # 文件路径遍历
        if '../' in content:
            issues.append("可能存在路径遍历风险")
        
        return issues
    
    async def _detect_performance_issues(self, content: str) -> List[str]:
        """检测性能问题"""
        issues = []
        
        # 循环中的数据库查询
        if re.search(r'for.*in.*:.*\.query\(', content):
            issues.append("循环中可能存在数据库查询")
        
        # 大文件一次性读取
        if 'file.read()' in content and 'with open' in content:
            issues.append("可能存在大文件一次性读取")
        
        # 低效字符串操作
        if content.count('+') > 50 and 'str' in content:
            issues.append("可能存在低效字符串操作")
        
        # 未使用缓存
        if 'database' in content.lower() and 'cache' not in content.lower():
            issues.append("数据库操作未使用缓存")
        
        return issues
    
    async def _detect_code_quality_issues(self, content: str) -> List[str]:
        """检测代码质量问题"""
        issues = []
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # 行长度
            if len(line) > 120:
                issues.append(f"第{i}行过长 ({len(line)}字符)")
            
            # TODO注释
            if 'TODO' in line or 'FIXME' in line:
                issues.append(f"第{i}行有待办事项")
            
            # 调试代码
            if 'print(' in line and 'debug' not in line.lower():
                issues.append(f"第{i}行可能有调试代码")
            
            # 空函数/类
            if 'def ' in line and 'pass' in line:
                issues.append(f"第{i}行有空函数")
        
        return issues
    
    async def _calculate_optimization_potential(self, security_issues: List[str], 
                                               performance_issues: List[str], 
                                               code_quality_issues: List[str]) -> float:
        """计算优化潜力"""
        potential = 0.0
        
        # 安全问题权重高
        potential += len(security_issues) * 0.3
        
        # 性能问题权重中等
        potential += len(performance_issues) * 0.2
        
        # 代码质量问题权重低
        potential += len(code_quality_issues) * 0.1
        
        return min(potential, 1.0)
    
    async def _analyze_functionality(self, content: str, filename: str) -> str:
        """分析功能描述"""
        # 基于文件名和内容分析功能
        if 'engine' in filename.lower():
            return "核心引擎模块，负责系统的主要功能实现"
        elif 'cache' in filename.lower():
            return "缓存系统模块，提供数据缓存和性能优化功能"
        elif 'security' in filename.lower():
            return "安全模块，负责系统安全防护和权限控制"
        elif 'test' in filename.lower():
            return "测试模块，确保系统功能正确性和稳定性"
        elif 'workflow' in filename.lower():
            return "工作流模块，管理和协调业务流程"
        elif 'api' in filename.lower():
            return "API接口模块，提供外部接口服务"
        elif 'util' in filename.lower():
            return "工具模块，提供通用工具和辅助功能"
        elif 'config' in filename.lower():
            return "配置模块，管理系统配置和参数"
        else:
            return "通用功能模块，提供特定的业务功能"
    
    async def _analyze_advantages_disadvantages(self, content: str, filename: str) -> Tuple[List[str], List[str]]:
        """分析优缺点"""
        advantages = []
        disadvantages = []
        
        # 基于文件类型分析
        if 'engine' in filename.lower():
            advantages.append("核心功能实现")
            advantages.append("高性能处理")
            disadvantages.append("复杂度高")
            disadvantages.append("维护成本高")
        
        elif 'cache' in filename.lower():
            advantages.append("提升性能")
            advantages.append("减少重复计算")
            disadvantages.append("内存占用")
            disadvantages.append("数据一致性挑战")
        
        elif 'test' in filename.lower():
            advantages.append("保证代码质量")
            advantages.append("防止回归错误")
            disadvantages.append("需要维护")
            disadvantages.append("执行时间开销")
        
        # 基于内容分析
        if 'class' in content:
            advantages.append("面向对象设计")
        
        if 'async def' in content:
            advantages.append("异步处理能力")
            disadvantages.append("调试复杂度增加")
        
        if len(content) > 1000:
            disadvantages.append("代码量较大")
        
        if 'import' in content:
            advantages.append("模块化设计")
            disadvantages.append("外部依赖")
        
        return advantages, disadvantages
    
    async def _generate_retention_justification(self, functionality: str, 
                                              advantages: List[str], 
                                              disadvantages: List[str]) -> str:
        """生成保留理由"""
        justification = f"功能描述：{functionality}\n\n"
        
        if advantages:
            justification += "优势：\n"
            for advantage in advantages:
                justification += f"- {advantage}\n"
        
        if disadvantages:
            justification += "\n劣势：\n"
            for disadvantage in disadvantages:
                justification += f"- {disadvantage}\n"
        
        justification += f"\n保留理由：该模块提供了{functionality}，"
        
        if len(advantages) > len(disadvantages):
            justification += "优势大于劣势，对系统有重要价值。"
        else:
            justification += "虽然有不足，但功能不可替代，需要保留。"
        
        return justification
    
    async def _generate_removal_justification(self, functionality: str, 
                                            disadvantages: List[str], 
                                            security_issues: List[str]) -> Optional[str]:
        """生成删除理由"""
        if not security_issues and len(disadvantages) < 3:
            return None
        
        justification = f"删除理由：该模块({functionality})"
        
        if security_issues:
            justification += f"存在{len(security_issues)}个安全问题，"
        
        if len(disadvantages) > 2:
            justification += f"有{len(disadvantages)}个主要缺点，"
        
        justification += "维护成本高且功能可被替代。"
        
        return justification
    
    async def _classify_and_prioritize_issues(self, file_analyses: List[FileAnalysisResult]) -> List[OptimizationItem]:
        """分类和优先级排序问题"""
        optimization_items = []
        
        for analysis in file_analyses:
            # 安全问题
            for issue in analysis.security_issues:
                item = OptimizationItem(
                    file_path=analysis.file_path,
                    issue_type="security",
                    severity="high",
                    description=issue,
                    evidence=[f"文件: {analysis.file_path}"],
                    recommendation="立即修复安全问题",
                    impact_assessment="高",
                    implementation_effort="中等",
                    priority_score=0.9,
                    dependencies=[]
                )
                optimization_items.append(item)
            
            # 性能问题
            for issue in analysis.performance_issues:
                item = OptimizationItem(
                    file_path=analysis.file_path,
                    issue_type="performance",
                    severity="medium",
                    description=issue,
                    evidence=[f"文件: {analysis.file_path}"],
                    recommendation="优化性能瓶颈",
                    impact_assessment="中等",
                    implementation_effort="中等",
                    priority_score=0.7,
                    dependencies=[]
                )
                optimization_items.append(item)
            
            # 代码质量问题
            for issue in analysis.code_quality_issues:
                item = OptimizationItem(
                    file_path=analysis.file_path,
                    issue_type="code_quality",
                    severity="low",
                    description=issue,
                    evidence=[f"文件: {analysis.file_path}"],
                    recommendation="改进代码质量",
                    impact_assessment="低",
                    implementation_effort="低",
                    priority_score=0.5,
                    dependencies=[]
                )
                optimization_items.append(item)
            
            # 文件级别优化建议
            if analysis.optimization_potential > 0.5:
                item = OptimizationItem(
                    file_path=analysis.file_path,
                    issue_type="file_optimization",
                    severity="medium",
                    description=f"文件优化潜力: {analysis.optimization_potential:.2f}",
                    evidence=[
                        f"复杂度: {analysis.complexity_score:.2f}",
                        f"可维护性: {analysis.maintainability_index:.2f}",
                        f"重复率: {analysis.duplication_ratio:.2f}"
                    ],
                    recommendation="重构文件以提升质量",
                    impact_assessment="中等",
                    implementation_effort="高",
                    priority_score=analysis.optimization_potential,
                    dependencies=[]
                )
                optimization_items.append(item)
        
        # 按优先级排序
        optimization_items.sort(key=lambda x: x.priority_score, reverse=True)
        
        return optimization_items
    
    async def _generate_optimization_strategies(self, optimization_items: List[OptimizationItem]) -> Dict[str, Any]:
        """生成优化策略"""
        strategies = {
            "immediate_actions": [],
            "short_term_goals": [],
            "long_term_plans": [],
            "resource_requirements": {},
            "risk_mitigation": []
        }
        
        # 按严重程度分类
        high_priority = [item for item in optimization_items if item.severity == "high"]
        medium_priority = [item for item in optimization_items if item.severity == "medium"]
        low_priority = [item for item in optimization_items if item.severity == "low"]
        
        # 立即行动项
        if high_priority:
            strategies["immediate_actions"].append({
                "action": "修复所有高严重性问题",
                "items_count": len(high_priority),
                "estimated_effort": "高",
                "impact": "显著提升系统安全性和稳定性"
            })
        
        # 短期目标
        if medium_priority:
            strategies["short_term_goals"].append({
                "goal": "优化性能和代码质量",
                "items_count": len(medium_priority),
                "estimated_effort": "中等",
                "impact": "提升系统性能和可维护性"
            })
        
        # 长期计划
        if low_priority:
            strategies["long_term_plans"].append({
                "plan": "持续改进和重构",
                "items_count": len(low_priority),
                "estimated_effort": "持续",
                "impact": "保持代码质量和技术债务控制"
            })
        
        # 资源需求
        total_items = len(optimization_items)
        strategies["resource_requirements"] = {
            "developer_days": total_items * 0.5,  # 估算
            "testing_days": total_items * 0.2,
            "review_days": total_items * 0.1
        }
        
        # 风险缓解
        strategies["risk_mitigation"] = [
            "分阶段实施，降低风险",
            "充分测试，确保功能正常",
            "备份代码，支持快速回滚",
            "团队协作，交叉审查"
        ]
        
        return strategies
    
    async def _assess_optimization_impact(self, optimization_items: List[OptimizationItem]) -> Dict[str, Any]:
        """评估优化影响"""
        impact = {
            "security_improvement": 0,
            "performance_improvement": 0,
            "code_quality_improvement": 0,
            "maintainability_improvement": 0,
            "overall_benefit": 0
        }
        
        for item in optimization_items:
            if item.issue_type == "security":
                impact["security_improvement"] += item.priority_score
            elif item.issue_type == "performance":
                impact["performance_improvement"] += item.priority_score
            elif item.issue_type == "code_quality":
                impact["code_quality_improvement"] += item.priority_score
            elif item.issue_type == "file_optimization":
                impact["maintainability_improvement"] += item.priority_score
        
        # 计算整体收益
        impact["overall_benefit"] = (
            impact["security_improvement"] * 0.4 +
            impact["performance_improvement"] * 0.3 +
            impact["code_quality_improvement"] * 0.2 +
            impact["maintainability_improvement"] * 0.1
        )
        
        return impact
    
    async def _create_implementation_plan(self, optimization_items: List[OptimizationItem]) -> Dict[str, Any]:
        """创建实施计划"""
        plan = {
            "phases": [],
            "timeline": {},
            "milestones": [],
            "success_criteria": []
        }
        
        # 分阶段计划
        total_items = len(optimization_items)
        
        # 第一阶段：高优先级问题
        high_priority_items = [item for item in optimization_items if item.severity == "high"]
        if high_priority_items:
            plan["phases"].append({
                "phase": 1,
                "name": "紧急修复",
                "duration": "1-2周",
                "items": high_priority_items[:10],  # 限制数量
                "focus": "安全问题和高优先级问题"
            })
        
        # 第二阶段：性能优化
        performance_items = [item for item in optimization_items if item.issue_type == "performance"]
        if performance_items:
            plan["phases"].append({
                "phase": 2,
                "name": "性能优化",
                "duration": "2-3周",
                "items": performance_items[:10],
                "focus": "性能瓶颈优化"
            })
        
        # 第三阶段：代码质量提升
        quality_items = [item for item in optimization_items if item.issue_type == "code_quality"]
        if quality_items:
            plan["phases"].append({
                "phase": 3,
                "name": "代码质量提升",
                "duration": "3-4周",
                "items": quality_items[:20],
                "focus": "代码质量和可维护性"
            })
        
        # 时间线
        plan["timeline"] = {
            "total_duration": f"{len(plan['phases']) * 2}-{len(plan['phases']) * 3}周",
            "start_date": datetime.now().strftime("%Y-%m-%d"),
            "estimated_completion": "基于阶段持续时间计算"
        }
        
        # 里程碑
        for phase in plan["phases"]:
            plan["milestones"].append({
                "milestone": f"{phase['name']}完成",
                "criteria": f"所有{phase['name']}项目完成并测试通过",
                "deliverables": f"{phase['name']}报告和代码更新"
            })
        
        # 成功标准
        plan["success_criteria"] = [
            "所有高严重性问题已解决",
            "性能指标提升20%以上",
            "代码质量评分提升至80分以上",
            "测试覆盖率提升至30%以上",
            "系统稳定性显著改善"
        ]
        
        return plan
    
    async def _create_final_report(self, structure_analysis: Dict[str, Any],
                                 file_analyses: List[FileAnalysisResult],
                                 optimization_items: List[OptimizationItem],
                                 optimization_strategies: Dict[str, Any],
                                 impact_assessment: Dict[str, Any],
                                 implementation_plan: Dict[str, Any]) -> Dict[str, Any]:
        """创建最终报告"""
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "project_root": str(self.project_root),
                "report_version": "1.0",
                "analysis_scope": "comprehensive"
            },
            "executive_summary": {
                "total_files_analyzed": len(file_analyses),
                "total_optimization_items": len(optimization_items),
                "high_priority_items": len([item for item in optimization_items if item.severity == "high"]),
                "estimated_effort": f"{len(optimization_items) * 0.5}人天",
                "overall_benefit_score": impact_assessment["overall_benefit"],
                "recommendation": "按计划分阶段实施优化"
            },
            "structure_analysis": structure_analysis,
            "file_analyses": [asdict(analysis) for analysis in file_analyses],
            "optimization_items": [asdict(item) for item in optimization_items],
            "optimization_strategies": optimization_strategies,
            "impact_assessment": impact_assessment,
            "implementation_plan": implementation_plan,
            "conclusions_and_next_steps": [
                "项目整体结构良好，但存在优化空间",
                "安全问题需要优先处理",
                "性能优化可以显著提升用户体验",
                "代码质量改进有助于长期维护",
                "建议按照实施计划分阶段执行"
            ]
        }
        
        return report
    
    async def _save_report(self, report: Dict[str, Any]):
        """保存报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON报告
        json_file = self.reports_dir / f"optimization_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 保存Markdown报告
        markdown_file = self.reports_dir / f"optimization_report_{timestamp}.md"
        markdown_content = await self._generate_markdown_report(report)
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"📄 优化报告已保存:")
        print(f"  JSON: {json_file}")
        print(f"  Markdown: {markdown_file}")
    
    async def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """生成Markdown报告"""
        content = []
        
        # 标题
        content.append("# 项目优化报告")
        content.append(f"生成时间: {report['metadata']['generated_at']}")
        content.append("")
        
        # 执行摘要
        content.append("## 📊 执行摘要")
        summary = report["executive_summary"]
        content.append(f"- 分析文件总数: {summary['total_files_analyzed']}")
        content.append(f"- 优化项总数: {summary['total_optimization_items']}")
        content.append(f"- 高优先级项: {summary['high_priority_items']}")
        content.append(f"- 预估工作量: {summary['estimated_effort']}")
        content.append(f"- 整体收益评分: {summary['overall_benefit_score']:.2f}")
        content.append(f"- 总体建议: {summary['recommendation']}")
        content.append("")
        
        # 结构分析
        content.append("## 🏗️ 项目结构分析")
        structure = report["structure_analysis"]
        content.append(f"- 总文件数: {structure['total_files']}")
        content.append(f"- Python文件: {structure['python_files']}")
        content.append(f"- 测试文件: {structure['test_files']}")
        content.append(f"- 配置文件: {structure['config_files']}")
        content.append(f"- 文档文件: {structure['doc_files']}")
        content.append("")
        
        # 优化项统计
        content.append("## 🎯 优化项统计")
        items = report["optimization_items"]
        
        # 按类型统计
        type_counts = {}
        severity_counts = {}
        
        for item in items:
            type_counts[item["issue_type"]] = type_counts.get(item["issue_type"], 0) + 1
            severity_counts[item["severity"]] = severity_counts.get(item["severity"], 0) + 1
        
        content.append("### 按类型分类")
        for issue_type, count in type_counts.items():
            content.append(f"- {issue_type}: {count}个")
        
        content.append("\n### 按严重程度分类")
        for severity, count in severity_counts.items():
            content.append(f"- {severity}: {count}个")
        content.append("")
        
        # 高优先级项详情
        high_priority_items = [item for item in items if item["severity"] == "high"]
        if high_priority_items:
            content.append("## 🚨 高优先级优化项")
            for item in high_priority_items[:10]:  # 限制显示数量
                content.append(f"### {item['file_path']}")
                content.append(f"- **问题**: {item['description']}")
                content.append(f"- **建议**: {item['recommendation']}")
                content.append(f"- **影响**: {item['impact_assessment']}")
                content.append(f"- **优先级**: {item['priority_score']:.2f}")
                content.append("")
        
        # 实施计划
        content.append("## 📋 实施计划")
        plan = report["implementation_plan"]
        
        content.append("### 阶段规划")
        for phase in plan["phases"]:
            content.append(f"#### 阶段{phase['phase']}: {phase['name']}")
            content.append(f"- 持续时间: {phase['duration']}")
            content.append(f"- 重点关注: {phase['focus']}")
            content.append(f"- 项目数量: {len(phase['items'])}")
            content.append("")
        
        # 成功标准
        content.append("### 成功标准")
        for criteria in plan["success_criteria"]:
            content.append(f"- {criteria}")
        content.append("")
        
        # 结论
        content.append("## 🎯 结论和下一步")
        for conclusion in report["conclusions_and_next_steps"]:
            content.append(f"- {conclusion}")
        content.append("")
        
        return "\n".join(content)

# 使用示例
async def main():
    """主函数"""
    project_root = "."
    
    generator = OptimizationReportGenerator(project_root)
    report = await generator.generate_comprehensive_report()
    
    print("🎉 优化报告生成完成!")
    print(f"📊 分析了 {report['executive_summary']['total_files_analyzed']} 个文件")
    print(f"🎯 发现了 {report['executive_summary']['total_optimization_items']} 个优化项")
    print(f"⚠️ 高优先级项: {report['executive_summary']['high_priority_items']} 个")

if __name__ == "__main__":
    asyncio.run(main())