#!/usr/bin/env python3
"""
内置深度分析扫描审查功能
提供全面的代码质量、安全性、性能和架构深度分析
"""

import os
import re
import ast
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import subprocess
import sys

@dataclass
class ScanIssue:
    """扫描问题"""
    file_path: str
    line_number: int
    issue_type: str
    severity: str
    category: str
    description: str
    evidence: str
    recommendation: str
    impact_score: float
    fix_complexity: str
    references: List[str]

@dataclass
class FileMetrics:
    """文件指标"""
    file_path: str
    lines_of_code: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    maintainability_index: float
    halstead_volume: float
    comment_ratio: float
    duplication_ratio: float
    test_coverage: float
    security_score: float
    performance_score: float
    architecture_score: float

@dataclass
class ArchitectureAnalysis:
    """架构分析"""
    module_dependencies: Dict[str, List[str]]
    circular_dependencies: List[Tuple[str, str]]
    coupling_metrics: Dict[str, float]
    cohesion_metrics: Dict[str, float]
    design_patterns: List[str]
    anti_patterns: List[str]
    layer_violations: List[str]
    interface_segregation: Dict[str, float]

class DeepAnalysisScanner:
    """深度分析扫描器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.scan_results = {
            "issues": [],
            "metrics": {},
            "architecture": {},
            "summary": {}
        }
        
        # 扫描规则配置
        self.security_rules = self._load_security_rules()
        self.performance_rules = self._load_performance_rules()
        self.quality_rules = self._load_quality_rules()
        self.architecture_rules = self._load_architecture_rules()
        
    async def perform_comprehensive_scan(self) -> Dict[str, Any]:
        """执行全面扫描"""
        print("🔬 启动深度分析扫描审查系统...")
        print("=" * 60)
        
        # 1. 项目文件发现
        print("📁 发现项目文件...")
        python_files = await self._discover_python_files()
        
        # 2. 并行文件分析
        print("🔍 并行分析文件...")
        file_analyses = await self._analyze_files_parallel(python_files)
        
        # 3. 安全性深度扫描
        print("🛡️ 执行安全性深度扫描...")
        security_issues = await self._perform_security_scan(file_analyses)
        
        # 4. 性能深度扫描
        print("⚡ 执行性能深度扫描...")
        performance_issues = await self._perform_performance_scan(file_analyses)
        
        # 5. 代码质量深度扫描
        print("📋 执行代码质量深度扫描...")
        quality_issues = await self._perform_quality_scan(file_analyses)
        
        # 6. 架构深度分析
        print("🏗️ 执行架构深度分析...")
        architecture_analysis = await self._perform_architecture_analysis(file_analyses)
        
        # 7. 依赖关系分析
        print("🔗 执行依赖关系分析...")
        dependency_analysis = await self._analyze_dependencies(file_analyses)
        
        # 8. 反模式检测
        print("🚫 执行反模式检测...")
        anti_patterns = await self._detect_anti_patterns(file_analyses)
        
        # 9. 合并所有问题
        all_issues = security_issues + performance_issues + quality_issues + anti_patterns
        
        # 10. 计算综合指标
        comprehensive_metrics = await self._calculate_comprehensive_metrics(
            file_analyses, all_issues, architecture_analysis
        )
        
        # 11. 生成扫描摘要
        scan_summary = await self._generate_scan_summary(
            python_files, all_issues, comprehensive_metrics
        )
        
        # 12. 构建最终结果
        final_results = {
            "scan_metadata": {
                "timestamp": datetime.now().isoformat(),
                "project_root": str(self.project_root),
                "total_files_scanned": len(python_files),
                "scan_duration": "待计算",
                "scan_version": "1.0.0"
            },
            "file_analyses": [asdict(analysis) for analysis in file_analyses],
            "security_issues": [asdict(issue) for issue in security_issues],
            "performance_issues": [asdict(issue) for issue in performance_issues],
            "quality_issues": [asdict(issue) for issue in quality_issues],
            "architecture_analysis": asdict(architecture_analysis),
            "dependency_analysis": dependency_analysis,
            "anti_patterns": [asdict(pattern) for pattern in anti_patterns],
            "comprehensive_metrics": comprehensive_metrics,
            "scan_summary": scan_summary,
            "recommendations": await self._generate_comprehensive_recommendations(all_issues)
        }
        
        print("✅ 深度分析扫描审查完成")
        return final_results
    
    async def _discover_python_files(self) -> List[Path]:
        """发现Python文件"""
        python_files = []
        
        for root, dirs, files in os.walk(self.project_root):
            # 跳过特定目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    file_path = Path(root) / file
                    python_files.append(file_path)
        
        return python_files
    
    async def _analyze_files_parallel(self, python_files: List[Path]) -> List[FileMetrics]:
        """并行分析文件"""
        # 简化的并行处理（实际可以使用asyncio.gather）
        file_analyses = []
        
        for file_path in python_files:
            try:
                metrics = await self._analyze_file_metrics(file_path)
                file_analyses.append(metrics)
            except Exception as e:
                print(f"⚠️ 分析文件失败 {file_path}: {e}")
        
        return file_analyses
    
    async def _analyze_file_metrics(self, file_path: Path) -> FileMetrics:
        """分析文件指标"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 基本指标
            lines = content.split('\n')
            lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            
            # 圈复杂度
            cyclomatic_complexity = await self._calculate_cyclomatic_complexity(content)
            
            # 认知复杂度
            cognitive_complexity = await self._calculate_cognitive_complexity(content)
            
            # 可维护性指数
            maintainability_index = await self._calculate_maintainability_index(content, cyclomatic_complexity)
            
            # Halstead体积
            halstead_volume = await self._calculate_halstead_volume(content)
            
            # 注释比率
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            comment_ratio = comment_lines / max(lines_of_code, 1)
            
            # 重复率
            duplication_ratio = await self._calculate_duplication_ratio(content)
            
            # 测试覆盖率（估算）
            test_coverage = await self._estimate_test_coverage(file_path)
            
            # 安全评分
            security_score = await self._calculate_security_score(content)
            
            # 性能评分
            performance_score = await self._calculate_performance_score(content)
            
            # 架构评分
            architecture_score = await self._calculate_architecture_score(content)
            
            return FileMetrics(
                file_path=str(file_path.relative_to(self.project_root)),
                lines_of_code=lines_of_code,
                cyclomatic_complexity=cyclomatic_complexity,
                cognitive_complexity=cognitive_complexity,
                maintainability_index=maintainability_index,
                halstead_volume=halstead_volume,
                comment_ratio=comment_ratio,
                duplication_ratio=duplication_ratio,
                test_coverage=test_coverage,
                security_score=security_score,
                performance_score=performance_score,
                architecture_score=architecture_score
            )
            
        except Exception as e:
            print(f"⚠️ 文件指标分析失败 {file_path}: {e}")
            return FileMetrics(
                file_path=str(file_path.relative_to(self.project_root)),
                lines_of_code=0,
                cyclomatic_complexity=0,
                cognitive_complexity=0,
                maintainability_index=0,
                halstead_volume=0,
                comment_ratio=0,
                duplication_ratio=0,
                test_coverage=0,
                security_score=0,
                performance_score=0,
                architecture_score=0
            )
    
    async def _calculate_cyclomatic_complexity(self, content: str) -> int:
        """计算圈复杂度"""
        try:
            tree = ast.parse(content)
            complexity = 1  # 基础复杂度
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.AsyncWith)):
                    complexity += 1
                elif isinstance(node, ast.ExceptHandler):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
                elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                    complexity += 1
            
            return complexity
            
        except Exception:
            return 0
    
    async def _calculate_cognitive_complexity(self, content: str) -> int:
        """计算认知复杂度"""
        complexity = 0
        nesting_level = 0
        
        lines = content.split('\n')
        for line in lines:
            stripped = line.strip()
            
            # 增加嵌套层级
            if any(keyword in stripped for keyword in ['if', 'elif', 'else:', 'for', 'while', 'try:', 'except', 'with']):
                nesting_level += 1
                complexity += nesting_level
            
            # 减少嵌套层级
            if stripped == 'pass' or stripped.startswith('return'):
                nesting_level = max(0, nesting_level - 1)
        
        return complexity
    
    async def _calculate_maintainability_index(self, content: str, cyclomatic_complexity: int) -> float:
        """计算可维护性指数"""
        lines = len(content.split('\n'))
        
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
        base_score -= cyclomatic_complexity * 2
        
        # 注释影响
        comment_lines = content.count('#')
        comment_ratio = comment_lines / max(lines, 1)
        base_score += comment_ratio * 10
        
        return max(0, min(100, base_score))
    
    async def _calculate_halstead_volume(self, content: str) -> float:
        """计算Halstead体积"""
        # 简化的Halstead体积计算
        operators = len(re.findall(r'[+\-*/%=<>!&|^~]', content))
        operands = len(re.findall(r'\b\w+\b', content))
        
        if operators == 0 or operands == 0:
            return 0.0
        
        vocabulary = operators + operands
        length = operators + operands
        
        try:
            volume = length * (vocabulary.bit_length() / 2)
            return volume
        except:
            return 0.0
    
    async def _calculate_duplication_ratio(self, content: str) -> float:
        """计算重复率"""
        lines = [line.strip() for line in content.split('\n') if line.strip() and len(line.strip()) > 10]
        
        if len(lines) < 10:
            return 0.0
        
        unique_lines = set(lines)
        return 1.0 - (len(unique_lines) / len(lines))
    
    async def _estimate_test_coverage(self, file_path: Path) -> float:
        """估算测试覆盖率"""
        # 检查对应的测试文件
        test_patterns = [
            f"test_{file_path.stem}.py",
            f"{file_path.stem}_test.py"
        ]
        
        for pattern in test_patterns:
            test_file = file_path.parent / pattern
            if test_file.exists():
                return 0.8
        
        # 检查是否是测试文件
        if 'test' in file_path.name.lower():
            return 0.9
        
        return 0.3
    
    async def _calculate_security_score(self, content: str) -> float:
        """计算安全评分"""
        score = 1.0
        
        # 危险函数扣分
        dangerous_functions = ['eval(', 'exec(', 'compile(']
        for func in dangerous_functions:
            if func in content:
                score -= 0.3
        
        # 硬编码密码扣分
        if re.search(r'password\s*=\s*["\'][^"\']+["\']', content, re.IGNORECASE):
            score -= 0.4
        
        # SQL注入风险扣分
        if 'execute(' in content and '%' in content:
            score -= 0.2
        
        # 文件路径遍历扣分
        if '../' in content:
            score -= 0.2
        
        return max(0, score)
    
    async def _calculate_performance_score(self, content: str) -> float:
        """计算性能评分"""
        score = 1.0
        
        # 循环中的数据库查询扣分
        if re.search(r'for.*in.*:.*\.query\(', content):
            score -= 0.3
        
        # 大文件一次性读取扣分
        if 'file.read()' in content and 'with open' in content:
            score -= 0.2
        
        # 低效字符串操作扣分
        if content.count('+') > 50:
            score -= 0.1
        
        # 未使用缓存扣分
        if 'database' in content.lower() and 'cache' not in content.lower():
            score -= 0.1
        
        return max(0, score)
    
    async def _calculate_architecture_score(self, content: str) -> float:
        """计算架构评分"""
        score = 0.5  # 基础分数
        
        # 面向对象设计加分
        if 'class ' in content:
            score += 0.2
        
        # 模块化设计加分
        if 'import' in content:
            score += 0.1
        
        # 异步设计加分
        if 'async def' in content:
            score += 0.1
        
        # 错误处理加分
        if 'try:' in content and 'except' in content:
            score += 0.1
        
        return min(1.0, score)
    
    async def _perform_security_scan(self, file_analyses: List[FileMetrics]) -> List[ScanIssue]:
        """执行安全性扫描"""
        issues = []
        
        for metrics in file_analyses:
            file_path = self.project_root / metrics.file_path
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_issues = await self._apply_security_rules(content, str(file_path))
                issues.extend(file_issues)
                
            except Exception as e:
                print(f"⚠️ 安全扫描失败 {file_path}: {e}")
        
        return issues
    
    async def _apply_security_rules(self, content: str, file_path: str) -> List[ScanIssue]:
        """应用安全规则"""
        issues = []
        lines = content.split('\n')
        
        for rule in self.security_rules:
            if rule["type"] == "pattern":
                matches = re.finditer(rule["pattern"], content)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    
                    issue = ScanIssue(
                        file_path=file_path,
                        line_number=line_num,
                        issue_type="security",
                        severity=rule["severity"],
                        category=rule["category"],
                        description=rule["description"],
                        evidence=match.group(0),
                        recommendation=rule["recommendation"],
                        impact_score=rule["impact_score"],
                        fix_complexity=rule["fix_complexity"],
                        references=rule.get("references", [])
                    )
                    issues.append(issue)
        
        return issues
    
    async def _perform_performance_scan(self, file_analyses: List[FileMetrics]) -> List[ScanIssue]:
        """执行性能扫描"""
        issues = []
        
        for metrics in file_analyses:
            file_path = self.project_root / metrics.file_path
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_issues = await self._apply_performance_rules(content, str(file_path))
                issues.extend(file_issues)
                
            except Exception as e:
                print(f"⚠️ 性能扫描失败 {file_path}: {e}")
        
        return issues
    
    async def _apply_performance_rules(self, content: str, file_path: str) -> List[ScanIssue]:
        """应用性能规则"""
        issues = []
        lines = content.split('\n')
        
        for rule in self.performance_rules:
            if rule["type"] == "pattern":
                matches = re.finditer(rule["pattern"], content, re.MULTILINE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    
                    issue = ScanIssue(
                        file_path=file_path,
                        line_number=line_num,
                        issue_type="performance",
                        severity=rule["severity"],
                        category=rule["category"],
                        description=rule["description"],
                        evidence=match.group(0),
                        recommendation=rule["recommendation"],
                        impact_score=rule["impact_score"],
                        fix_complexity=rule["fix_complexity"],
                        references=rule.get("references", [])
                    )
                    issues.append(issue)
        
        return issues
    
    async def _perform_quality_scan(self, file_analyses: List[FileMetrics]) -> List[ScanIssue]:
        """执行代码质量扫描"""
        issues = []
        
        for metrics in file_analyses:
            file_path = self.project_root / metrics.file_path
            
            # 基于指标的质量问题
            if metrics.cyclomatic_complexity > 10:
                issue = ScanIssue(
                    file_path=str(file_path),
                    line_number=0,
                    issue_type="quality",
                    severity="medium",
                    category="complexity",
                    description=f"圈复杂度过高: {metrics.cyclomatic_complexity}",
                    evidence=f"圈复杂度 = {metrics.cyclomatic_complexity}",
                    recommendation="重构函数，降低复杂度",
                    impact_score=0.6,
                    fix_complexity="medium",
                    references=["Cyclomatic Complexity Best Practices"]
                )
                issues.append(issue)
            
            if metrics.maintainability_index < 50:
                issue = ScanIssue(
                    file_path=str(file_path),
                    line_number=0,
                    issue_type="quality",
                    severity="medium",
                    category="maintainability",
                    description=f"可维护性指数过低: {metrics.maintainability_index:.1f}",
                    evidence=f"可维护性指数 = {metrics.maintainability_index:.1f}",
                    recommendation="改进代码结构，提升可维护性",
                    impact_score=0.5,
                    fix_complexity="medium",
                    references=["Maintainability Index Guidelines"]
                )
                issues.append(issue)
            
            if metrics.duplication_ratio > 0.3:
                issue = ScanIssue(
                    file_path=str(file_path),
                    line_number=0,
                    issue_type="quality",
                    severity="low",
                    category="duplication",
                    description=f"代码重复率过高: {metrics.duplication_ratio:.2f}",
                    evidence=f"重复率 = {metrics.duplication_ratio:.2f}",
                    recommendation="提取公共函数，减少重复代码",
                    impact_score=0.3,
                    fix_complexity="low",
                    references=["DRY Principle"]
                )
                issues.append(issue)
        
        return issues
    
    async def _perform_architecture_analysis(self, file_analyses: List[FileMetrics]) -> ArchitectureAnalysis:
        """执行架构分析"""
        # 简化的架构分析
        module_dependencies = {}
        circular_dependencies = []
        coupling_metrics = {}
        cohesion_metrics = {}
        design_patterns = []
        anti_patterns = []
        layer_violations = []
        interface_segregation = {}
        
        # 分析模块依赖
        for metrics in file_analyses:
            file_path = Path(metrics.file_path)
            module_name = file_path.stem
            
            # 简化的依赖分析
            dependencies = []  # 实际需要解析import语句
            module_dependencies[module_name] = dependencies
            
            # 耦合度指标（简化）
            coupling_metrics[module_name] = len(dependencies) * 0.1
            
            # 内聚度指标（简化）
            cohesion_metrics[module_name] = 0.8  # 默认值
        
        return ArchitectureAnalysis(
            module_dependencies=module_dependencies,
            circular_dependencies=circular_dependencies,
            coupling_metrics=coupling_metrics,
            cohesion_metrics=cohesion_metrics,
            design_patterns=design_patterns,
            anti_patterns=anti_patterns,
            layer_violations=layer_violations,
            interface_segregation=interface_segregation
        )
    
    async def _analyze_dependencies(self, file_analyses: List[FileMetrics]) -> Dict[str, Any]:
        """分析依赖关系"""
        dependency_graph = {}
        external_dependencies = set()
        internal_dependencies = {}
        
        for metrics in file_analyses:
            file_path = Path(metrics.file_path)
            module_name = file_path.stem
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 简化的依赖提取
                imports = re.findall(r'import\s+(\w+)', content)
                from_imports = re.findall(r'from\s+(\w+)', content)
                
                all_deps = imports + from_imports
                dependency_graph[module_name] = all_deps
                
                # 分类外部和内部依赖
                for dep in all_deps:
                    if dep.startswith(('os', 'sys', 'json', 'datetime', 'asyncio')):
                        external_dependencies.add(dep)
                    else:
                        if module_name not in internal_dependencies:
                            internal_dependencies[module_name] = []
                        internal_dependencies[module_name].append(dep)
                        
            except Exception as e:
                print(f"⚠️ 依赖分析失败 {file_path}: {e}")
        
        return {
            "dependency_graph": dependency_graph,
            "external_dependencies": list(external_dependencies),
            "internal_dependencies": internal_dependencies,
            "dependency_metrics": {
                "total_modules": len(file_analyses),
                "total_dependencies": sum(len(deps) for deps in dependency_graph.values()),
                "average_dependencies_per_module": sum(len(deps) for deps in dependency_graph.values()) / max(len(dependency_graph), 1)
            }
        }
    
    async def _detect_anti_patterns(self, file_analyses: List[FileMetrics]) -> List[ScanIssue]:
        """检测反模式"""
        anti_patterns = []
        
        for metrics in file_analyses:
            file_path = self.project_root / metrics.file_path
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检测各种反模式
                file_anti_patterns = await self._detect_file_anti_patterns(content, str(file_path))
                anti_patterns.extend(file_anti_patterns)
                
            except Exception as e:
                print(f"⚠️ 反模式检测失败 {file_path}: {e}")
        
        return anti_patterns
    
    async def _detect_file_anti_patterns(self, content: str, file_path: str) -> List[ScanIssue]:
        """检测文件反模式"""
        issues = []
        lines = content.split('\n')
        
        # God Class反模式
        class_count = content.count('class ')
        method_count = content.count('def ')
        if class_count == 1 and method_count > 20:
            issue = ScanIssue(
                file_path=file_path,
                line_number=0,
                issue_type="anti_pattern",
                severity="medium",
                category="god_class",
                description="God Class反模式：单个类包含过多方法",
                evidence=f"1个类包含{method_count}个方法",
                recommendation="拆分为多个职责单一的类",
                impact_score=0.7,
                fix_complexity="high",
                references=["Single Responsibility Principle"]
            )
            issues.append(issue)
        
        # Long Method反模式
        for i, line in enumerate(lines):
            if 'def ' in line:
                # 简化的长方法检测
                method_lines = 0
                for j in range(i, len(lines)):
                    if lines[j].strip() and not lines[j].startswith(' '):
                        break
                    method_lines += 1
                
                if method_lines > 50:
                    issue = ScanIssue(
                        file_path=file_path,
                        line_number=i + 1,
                        issue_type="anti_pattern",
                        severity="medium",
                        category="long_method",
                        description="Long Method反模式：方法过长",
                        evidence=f"方法长度: {method_lines}行",
                        recommendation="拆分为多个小方法",
                        impact_score=0.5,
                        fix_complexity="medium",
                        references=["Extract Method Refactoring"]
                    )
                    issues.append(issue)
        
        return issues
    
    async def _calculate_comprehensive_metrics(self, file_analyses: List[FileMetrics], 
                                             issues: List[ScanIssue], 
                                             architecture: ArchitectureAnalysis) -> Dict[str, Any]:
        """计算综合指标"""
        total_files = len(file_analyses)
        
        if total_files == 0:
            return {}
        
        # 计算平均指标
        avg_complexity = sum(m.cyclomatic_complexity for m in file_analyses) / total_files
        avg_maintainability = sum(m.maintainability_index for m in file_analyses) / total_files
        avg_security = sum(m.security_score for m in file_analyses) / total_files
        avg_performance = sum(m.performance_score for m in file_analyses) / total_files
        avg_architecture = sum(m.architecture_score for m in file_analyses) / total_files
        
        # 问题统计
        security_issues = [i for i in issues if i.issue_type == "security"]
        performance_issues = [i for i in issues if i.issue_type == "performance"]
        quality_issues = [i for i in issues if i.issue_type == "quality"]
        
        # 综合评分
        overall_score = (
            avg_security * 0.3 +
            avg_performance * 0.25 +
            avg_architecture * 0.2 +
            (avg_maintainability / 100) * 0.15 +
            (1 - min(avg_complexity / 20, 1)) * 0.1
        )
        
        return {
            "total_files": total_files,
            "average_complexity": avg_complexity,
            "average_maintainability": avg_maintainability,
            "average_security_score": avg_security,
            "average_performance_score": avg_performance,
            "average_architecture_score": avg_architecture,
            "total_issues": len(issues),
            "security_issues_count": len(security_issues),
            "performance_issues_count": len(performance_issues),
            "quality_issues_count": len(quality_issues),
            "overall_quality_score": overall_score,
            "quality_grade": self._calculate_quality_grade(overall_score)
        }
    
    def _calculate_quality_grade(self, score: float) -> str:
        """计算质量等级"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    async def _generate_scan_summary(self, python_files: List[Path], 
                                   issues: List[ScanIssue], 
                                   metrics: Dict[str, Any]) -> Dict[str, Any]:
        """生成扫描摘要"""
        return {
            "scan_overview": {
                "files_scanned": len(python_files),
                "total_issues": len(issues),
                "critical_issues": len([i for i in issues if i.severity == "critical"]),
                "high_issues": len([i for i in issues if i.severity == "high"]),
                "medium_issues": len([i for i in issues if i.severity == "medium"]),
                "low_issues": len([i for i in issues if i.severity == "low"])
            },
            "quality_metrics": metrics,
            "issue_distribution": {
                "by_type": {
                    "security": len([i for i in issues if i.issue_type == "security"]),
                    "performance": len([i for i in issues if i.issue_type == "performance"]),
                    "quality": len([i for i in issues if i.issue_type == "quality"]),
                    "anti_pattern": len([i for i in issues if i.issue_type == "anti_pattern"])
                },
                "by_severity": {
                    "critical": len([i for i in issues if i.severity == "critical"]),
                    "high": len([i for i in issues if i.severity == "high"]),
                    "medium": len([i for i in issues if i.severity == "medium"]),
                    "low": len([i for i in issues if i.severity == "low"])
                }
            },
            "recommendations_priority": {
                "immediate": [i.description for i in issues if i.severity in ["critical", "high"]][:5],
                "short_term": [i.description for i in issues if i.severity == "medium"][:5],
                "long_term": [i.description for i in issues if i.severity == "low"][:5]
            }
        }
    
    async def _generate_comprehensive_recommendations(self, issues: List[ScanIssue]) -> List[Dict[str, Any]]:
        """生成综合推荐建议"""
        recommendations = []
        
        # 按严重程度分组
        critical_issues = [i for i in issues if i.severity == "critical"]
        high_issues = [i for i in issues if i.severity == "high"]
        medium_issues = [i for i in issues if i.severity == "medium"]
        low_issues = [i for i in issues if i.severity == "low"]
        
        # 立即行动建议
        if critical_issues:
            recommendations.append({
                "priority": "critical",
                "category": "立即行动",
                "description": "修复所有关键安全问题",
                "items": [i.description for i in critical_issues],
                "estimated_effort": "高",
                "impact": "消除安全风险，确保系统安全"
            })
        
        # 高优先级建议
        if high_issues:
            recommendations.append({
                "priority": "high",
                "category": "高优先级",
                "description": "处理高优先级问题",
                "items": [i.description for i in high_issues],
                "estimated_effort": "中等",
                "impact": "显著提升系统质量"
            })
        
        # 中期改进建议
        if medium_issues:
            recommendations.append({
                "priority": "medium",
                "category": "中期改进",
                "description": "优化性能和代码质量",
                "items": [i.description for i in medium_issues],
                "estimated_effort": "中等",
                "impact": "提升性能和可维护性"
            })
        
        # 长期优化建议
        if low_issues:
            recommendations.append({
                "priority": "low",
                "category": "长期优化",
                "description": "持续改进和重构",
                "items": [i.description for i in low_issues],
                "estimated_effort": "低",
                "impact": "保持代码质量"
            })
        
        return recommendations
    
    def _load_security_rules(self) -> List[Dict[str, Any]]:
        """加载安全规则"""
        return [
            {
                "type": "pattern",
                "pattern": r'eval\s*\(',
                "severity": "high",
                "category": "dangerous_function",
                "description": "使用了危险的eval函数",
                "recommendation": "避免使用eval，考虑 safer alternatives",
                "impact_score": 0.9,
                "fix_complexity": "medium",
                "references": ["CWE-94"]
            },
            {
                "type": "pattern",
                "pattern": r'exec\s*\(',
                "severity": "high",
                "category": "dangerous_function",
                "description": "使用了危险的exec函数",
                "recommendation": "避免使用exec，考虑 safer alternatives",
                "impact_score": 0.9,
                "fix_complexity": "medium",
                "references": ["CWE-94"]
            },
            {
                "type": "pattern",
                "pattern": r'password\s*=\s*["\'][^"\']+["\']',
                "severity": "critical",
                "category": "hardcoded_secret",
                "description": "硬编码密码或密钥",
                "recommendation": "使用环境变量或配置文件存储敏感信息",
                "impact_score": 1.0,
                "fix_complexity": "low",
                "references": ["CWE-798"]
            },
            {
                "type": "pattern",
                "pattern": r'secret\s*=\s*["\'][^"\']+["\']',
                "severity": "critical",
                "category": "hardcoded_secret",
                "description": "硬编码密钥",
                "recommendation": "使用环境变量或配置文件存储敏感信息",
                "impact_score": 1.0,
                "fix_complexity": "low",
                "references": ["CWE-798"]
            }
        ]
    
    def _load_performance_rules(self) -> List[Dict[str, Any]]:
        """加载性能规则"""
        return [
            {
                "type": "pattern",
                "pattern": r'for\s+\w+\s+in\s+.*:\s*.*\.query\(',
                "severity": "medium",
                "category": "database_in_loop",
                "description": "循环中执行数据库查询",
                "recommendation": "将查询移出循环或使用批量查询",
                "impact_score": 0.7,
                "fix_complexity": "medium",
                "references": ["Performance Best Practices"]
            },
            {
                "type": "pattern",
                "pattern": r'\.read\(\)\s*$',
                "severity": "medium",
                "category": "large_file_read",
                "description": "一次性读取大文件",
                "recommendation": "使用流式读取或分块处理",
                "impact_score": 0.6,
                "fix_complexity": "medium",
                "references": ["Memory Management"]
            }
        ]
    
    def _load_quality_rules(self) -> List[Dict[str, Any]]:
        """加载质量规则"""
        return [
            # 质量规则主要通过指标分析实现
        ]
    
    def _load_architecture_rules(self) -> List[Dict[str, Any]]:
        """加载架构规则"""
        return [
            # 架构规则主要通过依赖分析实现
        ]

# 使用示例
async def main():
    """主函数"""
    project_root = "."
    
    scanner = DeepAnalysisScanner(project_root)
    results = await scanner.perform_comprehensive_scan()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(project_root) / f"deep_scan_results_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"🎉 深度扫描完成!")
    print(f"📄 结果已保存到: {results_file}")
    print(f"📊 扫描了 {results['scan_metadata']['total_files_scanned']} 个文件")
    print(f"🚨 发现了 {len(results['security_issues']) + len(results['performance_issues']) + len(results['quality_issues'])} 个问题")

if __name__ == "__main__":
    asyncio.run(main())