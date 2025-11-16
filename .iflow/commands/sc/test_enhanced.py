#!/usr/bin/env python3
"""
增强版 /sc:test 指令实现
提供全面的项目测试、分析和优化功能
"""

import os
import sys
import json
import time
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import difflib
import ast
import re

@dataclass
class TestConfiguration:
    """测试配置类"""
    project_root: str
    test_types: List[str]
    coverage_threshold: float = 25.0
    enable_security_scan: bool = True
    enable_performance_test: bool = True
    enable_deep_analysis: bool = True
    interactive_mode: bool = False
    force_ai_awareness: bool = True

@dataclass
class FileInfo:
    """文件信息类"""
    path: str
    size: int
    modified_time: float
    file_type: str
    functions: List[str]
    classes: List[str]
    imports: List[str]
    complexity_score: float
    dependencies: List[str]
    functionality_description: str
    advantages: List[str]
    disadvantages: List[str]
    retention_reason: Optional[str] = None
    duplicate_check: Optional[str] = None

@dataclass
class ProjectStructure:
    """项目结构类"""
    timestamp: str
    total_files: int
    total_dirs: int
    file_tree: Dict[str, Any]
    file_details: Dict[str, FileInfo]
    module_dependencies: Dict[str, List[str]]
    complexity_metrics: Dict[str, float]

class EnhancedTestEngine:
    """增强版测试引擎"""
    
    def __init__(self, config: TestConfiguration):
        self.config = config
        self.project_root = Path(config.project_root)
        self.test_results = {}
        self.project_structure_before = None
        self.project_structure_after = None
        self.optimization_report = {}
        
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行全面测试分析"""
        print("🚀 启动增强版 /sc:test 全面分析系统")
        print("=" * 60)
        
        # 1. 强制AI信息传递
        if self.config.force_ai_awareness:
            await self._force_ai_awareness()
        
        # 2. 获取项目结构（测试前）
        print("📊 分析项目结构（测试前）...")
        self.project_structure_before = await self._analyze_project_structure()
        
        # 3. 执行深度分析扫描审查
        print("🔍 执行深度分析扫描审查...")
        scan_results = await self._perform_deep_analysis_scan()
        
        # 4. 运行测试套件
        print("🧪 执行测试套件...")
        test_results = await self._run_test_suite()
        
        # 5. 安全扫描
        if self.config.enable_security_scan:
            print("🛡️ 执行安全扫描...")
            security_results = await self._perform_security_scan()
        else:
            security_results = {"status": "skipped"}
        
        # 6. 性能测试
        if self.config.enable_performance_test:
            print("⚡ 执行性能测试...")
            performance_results = await self._perform_performance_test()
        else:
            performance_results = {"status": "skipped"}
        
        # 7. 获取项目结构（测试后）
        print("📊 分析项目结构（测试后）...")
        self.project_structure_after = await self._analyze_project_structure()
        
        # 8. 生成结构对比分析
        print("🔄 生成项目结构对比分析...")
        structure_comparison = await self._compare_project_structures()
        
        # 9. 自动生成优化报告
        print("📈 自动生成优化报告...")
        self.optimization_report = await self._generate_optimization_report(
            test_results, security_results, performance_results, 
            scan_results, structure_comparison
        )
        
        # 10. 交互式处理
        if self.config.interactive_mode:
            await self._interactive_analysis()
        
        # 11. 生成最终报告
        final_report = await self._generate_final_report()
        
        print("✅ 增强版 /sc:test 分析完成！")
        return final_report
    
    async def _force_ai_awareness(self):
        """强制AI信息传递"""
        print("🤖 强制AI信息传递系统启动...")
        
        ai_context = {
            "project_name": "iFlow CLI V16 Quantum Evolution",
            "project_root": str(self.project_root),
            "timestamp": datetime.now().isoformat(),
            "test_objectives": [
                "全面测试覆盖分析",
                "深度代码质量审查",
                "安全性漏洞扫描",
                "性能基准测试",
                "项目结构优化分析",
                "文件功能特点评估",
                "保留/删除决策依据"
            ]
        }
    
    async def run_compatible_test(self, 
                                 target: Optional[str] = None,
                                 test_type: str = "all",
                                 enable_coverage: bool = True,
                                 watch_mode: bool = False,
                                 auto_fix: bool = False) -> Dict[str, Any]:
        """运行兼容模式测试（原始版本功能）"""
        print("🔄 运行兼容模式测试")
        print("=" * 60)
        
        # 1. 发现和配置测试
        print("🔍 发现测试配置...")
        test_config = await self._discover_test_configuration(target, test_type)
        
        # 2. 执行测试
        print("🧪 执行测试...")
        test_results = await self._run_test_suite(target, test_type, enable_coverage)
        
        # 3. 分析测试结果
        print("📊 分析测试结果...")
        analysis_results = await self._analyze_test_results(test_results)
        
        # 4. 生成报告
        print("📋 生成测试报告...")
        report = await self._generate_test_report(test_results, analysis_results)
        
        # 5. 处理监视模式
        if watch_mode:
            print("👁️ 启动监视模式...")
            await self._start_watch_mode(target, test_type, enable_coverage, auto_fix)
        
        # 6. 自动修复（如果启用）
        if auto_fix and test_results.get("failed", 0) > 0:
            print("🔧 尝试自动修复...")
            fix_results = await self._attempt_auto_fix(test_results)
            report["auto_fix_results"] = fix_results
        
        return {
            "mode": "compatible",
            "test_results": test_results,
            "analysis_results": analysis_results,
            "report": report
        }
    
    async def _discover_test_configuration(self, target: Optional[str], test_type: str) -> Dict[str, Any]:
        """发现测试配置（原始版本功能）"""
        config = {
            "test_framework": "pytest",
            "test_paths": [],
            "test_markers": [],
            "coverage_config": {}
        }
        
        # 查找测试文件
        if target:
            target_path = self.project_root / target
            if target_path.exists():
                config["test_paths"].append(str(target_path))
        else:
            # 自动发现测试目录
            test_dirs = ["tests", "test", "src/tests"]
            for test_dir in test_dirs:
                test_path = self.project_root / test_dir
                if test_path.exists():
                    config["test_paths"].append(str(test_path))
        
        # 设置测试标记
        if test_type == "unit":
            config["test_markers"] = ["unit"]
        elif test_type == "integration":
            config["test_markers"] = ["integration"]
        elif test_type == "e2e":
            config["test_markers"] = ["e2e"]
        
        # 检查配置文件
        config_files = ["pyproject.toml", "setup.cfg", "pytest.ini"]
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                config["config_file"] = str(config_path)
                break
        
        return config
    
    async def _analyze_test_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析测试结果（原始版本功能增强）"""
        analysis = {
            "success_rate": 0,
            "coverage_adequacy": "unknown",
            "performance_issues": [],
            "recommendations": []
        }
        
        total_tests = test_results.get("passed", 0) + test_results.get("failed", 0) + test_results.get("skipped", 0)
        if total_tests > 0:
            analysis["success_rate"] = (test_results.get("passed", 0) / total_tests) * 100
        
        # 分析覆盖率
        coverage = test_results.get("coverage", {})
        if coverage:
            coverage_pct = coverage.get("percent_covered", 0)
            if coverage_pct >= self.config.coverage_threshold:
                analysis["coverage_adequacy"] = "adequate"
            else:
                analysis["coverage_adequacy"] = "inadequate"
                analysis["recommendations"].append(
                    f"覆盖率 {coverage_pct:.1f}% 低于阈值 {self.config.coverage_threshold}%"
                )
        
        # 生成建议
        if test_results.get("failed", 0) > 0:
            analysis["recommendations"].append("有测试失败，请检查测试代码")
        
        if test_results.get("execution_time", 0) > 60:
            analysis["performance_issues"].append("测试执行时间过长")
        
        return analysis
    
    async def _generate_test_report(self, test_results: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成测试报告（原始版本功能）"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": test_results.get("passed", 0) + test_results.get("failed", 0) + test_results.get("skipped", 0),
                "passed": test_results.get("passed", 0),
                "failed": test_results.get("failed", 0),
                "skipped": test_results.get("skipped", 0),
                "success_rate": analysis["success_rate"],
                "execution_time": test_results.get("execution_time", 0)
            },
            "coverage": test_results.get("coverage", {}),
            "recommendations": analysis["recommendations"],
            "failure_analysis": test_results.get("failure_analysis")
        }
        
        # 保存报告
        report_file = self.project_root / "reports" / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 测试报告已保存: {report_file}")
        return report
    
    async def _start_watch_mode(self, target: Optional[str], test_type: str, enable_coverage: bool, auto_fix: bool):
        """启动监视模式（原始版本功能）"""
        print("👁️ 监视模式已启动，按 Ctrl+C 停止...")
        
        try:
            import time
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class TestFileHandler(FileSystemEventHandler):
                def __init__(self, callback):
                    self.callback = callback
                
                def on_modified(self, event):
                    if event.src_path.endswith('.py'):
                        print(f"📝 检测到文件变更: {event.src_path}")
                        self.callback()
            
            def run_tests():
                asyncio.create_task(
                    self._run_test_suite(target, test_type, enable_coverage)
                )
            
            observer = Observer()
            handler = TestFileHandler(run_tests)
            
            # 监视源代码目录
            watch_dirs = ["src", ".iflow"]
            for watch_dir in watch_dirs:
                watch_path = self.project_root / watch_dir
                if watch_path.exists():
                    observer.schedule(handler, str(watch_path), recursive=True)
            
            observer.start()
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            
            observer.join()
            
        except ImportError:
            print("⚠️ 需要安装 watchdog 库来使用监视模式: pip install watchdog")
    
    async def _attempt_auto_fix(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """尝试自动修复（原始版本功能）"""
        fix_results = {
            "attempted_fixes": 0,
            "successful_fixes": 0,
            "fix_details": []
        }
        
        # 简单的自动修复逻辑
        failure_analysis = test_results.get("failure_analysis", {})
        for pattern in failure_analysis.get("failure_patterns", []):
            if "ImportError" in pattern.get("error", ""):
                # 尝试修复导入错误
                fix_results["attempted_fixes"] += 1
                # 这里可以添加具体的修复逻辑
                fix_results["fix_details"].append({
                    "type": "import_error",
                    "target": pattern.get("module"),
                    "status": "identified"
                })
        
        return fix_results
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行全面测试分析"""
        analysis_config = {
            "critical_requirements": [
                "每一步都必须提供完整依据和解释",
                "所有文件决策都需要详细推理过程",
                "功能特点和优缺点必须明确列出",
                "删除文件必须有充分理由",
                "保留文件需要说明其独特价值"
            ],
            "project_structure": await self._get_basic_structure(),
            "test_configuration": asdict(self.config)
        }
        
        # 保存AI上下文到文件
        context_file = self.project_root / ".iflow" / "temp_docs" / "ai_context.json"
        context_file.parent.mkdir(exist_ok=True)
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(ai_context, f, ensure_ascii=False, indent=2)
        
        print(f"✅ AI上下文已保存到: {context_file}")
        print("🎯 AI已强制接收项目完整信息和测试要求")
    
    async def _analyze_project_structure(self) -> ProjectStructure:
        """分析项目结构"""
        structure = {
            "timestamp": datetime.now().isoformat(),
            "total_files": 0,
            "total_dirs": 0,
            "file_tree": {},
            "file_details": {},
            "module_dependencies": {},
            "complexity_metrics": {}
        }
        
        file_details = {}
        dependencies = {}
        complexity_metrics = {}
        
        # 遍历项目文件
        for root, dirs, files in os.walk(self.project_root):
            # 跳过特定目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            rel_root = os.path.relpath(root, self.project_root)
            if rel_root == '.':
                rel_root = 'root'
            
            structure["file_tree"][rel_root] = {
                "dirs": dirs.copy(),
                "files": files.copy()
            }
            
            structure["total_dirs"] += len(dirs)
            structure["total_files"] += len(files)
            
            # 分析每个文件
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    rel_path = str(file_path.relative_to(self.project_root))
                    
                    try:
                        file_info = await self._analyze_python_file(file_path)
                        file_details[rel_path] = file_info
                        
                        # 分析依赖关系
                        deps = await self._analyze_dependencies(file_path)
                        dependencies[rel_path] = deps
                        
                        # 计算复杂度
                        complexity = await self._calculate_complexity(file_path)
                        complexity_metrics[rel_path] = complexity
                        
                    except Exception as e:
                        print(f"⚠️ 分析文件失败 {rel_path}: {e}")
        
        structure["file_details"] = file_details
        structure["module_dependencies"] = dependencies
        structure["complexity_metrics"] = complexity_metrics
        
        return ProjectStructure(**structure)
    
    async def _analyze_python_file(self, file_path: Path) -> FileInfo:
        """分析Python文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # 提取函数和类
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        imports.extend([alias.name for alias in node.names])
                    else:
                        module = node.module or ""
                        imports.extend([f"{module}.{alias.name}" for alias in node.names])
            
            # 分析功能特点
            functionality = await self._analyze_functionality(content, file_path.name)
            
            # 计算复杂度分数
            complexity_score = len(functions) + len(classes) * 2 + len(imports) * 0.5
            
            return FileInfo(
                path=str(file_path.relative_to(self.project_root)),
                size=file_path.stat().st_size,
                modified_time=file_path.stat().st_mtime,
                file_type='python',
                functions=functions,
                classes=classes,
                imports=imports,
                complexity_score=complexity_score,
                dependencies=[],  # 将在后续填充
                functionality_description=functionality["description"],
                advantages=functionality["advantages"],
                disadvantages=functionality["disadvantages"]
            )
            
        except Exception as e:
            return FileInfo(
                path=str(file_path.relative_to(self.project_root)),
                size=file_path.stat().st_size,
                modified_time=file_path.stat().st_mtime,
                file_type='python',
                functions=[],
                classes=[],
                imports=[],
                complexity_score=0.0,
                dependencies=[],
                functionality_description=f"分析失败: {e}",
                advantages=[],
                disadvantages=["无法分析文件内容"]
            )
    
    async def _analyze_functionality(self, content: str, filename: str) -> Dict[str, Any]:
        """分析文件功能特点"""
        advantages = []
        disadvantages = []
        description = "通用Python模块"
        
        # 基于文件名和内容分析功能
        if 'test' in filename.lower():
            description = "测试模块"
            advantages.append("确保代码质量")
            advantages.append("防止回归错误")
            disadvantages.append("需要维护成本")
        
        elif 'engine' in filename.lower():
            description = "核心引擎模块"
            advantages.append("系统核心功能")
            advantages.append("高性能处理")
            disadvantages.append("复杂度高")
            disadvantages.append("依赖性强")
        
        elif 'cache' in filename.lower():
            description = "缓存系统模块"
            advantages.append("提升性能")
            advantages.append("减少重复计算")
            disadvantages.append("内存占用")
            disadvantages.append("数据一致性问题")
        
        elif 'security' in filename.lower():
            description = "安全相关模块"
            advantages.append("系统安全性")
            advantages.append("防护机制")
            disadvantages.append("性能开销")
            disadvantages.append("配置复杂")
        
        # 基于内容分析
        if 'class' in content and 'def' in content:
            advantages.append("面向对象设计")
        if 'async def' in content:
            advantages.append("异步处理能力")
            disadvantages.append("调试复杂度增加")
        if 'import' in content:
            advantages.append("模块化设计")
            disadvantages.append("外部依赖风险")
        
        return {
            "description": description,
            "advantages": advantages,
            "disadvantages": disadvantages
        }
    
    async def _analyze_dependencies(self, file_path: Path) -> List[str]:
        """分析文件依赖关系"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            dependencies = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        dependencies.append(f"{module}.{alias.name}")
            
            return dependencies
            
        except Exception:
            return []
    
    async def _calculate_complexity(self, file_path: Path) -> float:
        """计算文件复杂度"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            
            tree = ast.parse(content)
            
            # 计算圈复杂度
            complexity = 1  # 基础复杂度
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.With)):
                    complexity += 1
                elif isinstance(node, ast.ExceptHandler):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            return float(complexity + code_lines * 0.1)
            
        except Exception:
            return 0.0
    
    async def _perform_deep_analysis_scan(self) -> Dict[str, Any]:
        """执行深度分析扫描审查"""
        print("🔬 执行深度代码分析...")
        
        scan_results = {
            "timestamp": datetime.now().isoformat(),
            "total_files_scanned": 0,
            "issues_found": [],
            "recommendations": [],
            "code_quality_metrics": {},
            "duplicate_analysis": {},
            "unused_imports": {},
            "security_patterns": {},
            "performance_bottlenecks": []
        }
        
        # 扫描所有Python文件
        python_files = list(self.project_root.rglob("*.py"))
        scan_results["total_files_scanned"] = len(python_files)
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查代码质量问题
                issues = await self._check_code_quality(content, file_path)
                scan_results["issues_found"].extend(issues)
                
                # 检查重复代码
                duplicates = await self._check_duplicate_code(content, file_path)
                if duplicates:
                    scan_results["duplicate_analysis"][str(file_path.relative_to(self.project_root))] = duplicates
                
                # 检查未使用的导入
                unused_imports = await self._check_unused_imports(content, file_path)
                if unused_imports:
                    scan_results["unused_imports"][str(file_path.relative_to(self.project_root))] = unused_imports
                
                # 检查安全模式
                security_issues = await self._check_security_patterns(content, file_path)
                if security_issues:
                    scan_results["security_patterns"][str(file_path.relative_to(self.project_root))] = security_issues
                
                # 检查性能瓶颈
                perf_issues = await self._check_performance_bottlenecks(content, file_path)
                scan_results["performance_bottlenecks"].extend(perf_issues)
                
            except Exception as e:
                scan_results["issues_found"].append({
                    "file": str(file_path.relative_to(self.project_root)),
                    "type": "scan_error",
                    "message": f"扫描错误: {e}",
                    "severity": "medium"
                })
        
        # 生成推荐建议
        scan_results["recommendations"] = await self._generate_recommendations(scan_results)
        
        return scan_results
    
    async def _check_code_quality(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """检查代码质量"""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # 检查行长度
            if len(line) > 120:
                issues.append({
                    "file": str(file_path.relative_to(self.project_root)),
                    "line": i,
                    "type": "line_too_long",
                    "message": f"行长度超过120字符 ({len(line)}字符)",
                    "severity": "low"
                })
            
            # 检查TODO注释
            if 'TODO' in line or 'FIXME' in line:
                issues.append({
                    "file": str(file_path.relative_to(self.project_root)),
                    "line": i,
                    "type": "todo_comment",
                    "message": "存在待办事项注释",
                    "severity": "medium"
                })
            
            # 检查调试代码
            if 'print(' in line and 'debug' not in line.lower():
                issues.append({
                    "file": str(file_path.relative_to(self.project_root)),
                    "line": i,
                    "type": "debug_print",
                    "message": "可能存在调试代码",
                    "severity": "medium"
                })
        
        return issues
    
    async def _check_duplicate_code(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """检查重复代码"""
        duplicates = []
        
        # 简单的重复代码检测
        lines = content.split('\n')
        line_groups = {}
        
        for i, line in enumerate(lines):
            clean_line = line.strip()
            if len(clean_line) > 20:  # 只检查较长的行
                if clean_line not in line_groups:
                    line_groups[clean_line] = []
                line_groups[clean_line].append(i + 1)
        
        for line, line_numbers in line_groups.items():
            if len(line_numbers) > 1:
                duplicates.append({
                    "content": line,
                    "lines": line_numbers,
                    "type": "exact_duplicate"
                })
        
        return duplicates
    
    async def _check_unused_imports(self, content: str, file_path: Path) -> List[str]:
        """检查未使用的导入"""
        try:
            tree = ast.parse(content)
            imports = []
            
            # 获取所有导入
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            
            # 简单检查：如果导入在代码中没有出现，则认为未使用
            unused = []
            for imp in imports:
                name = imp.split('.')[-1]
                if name not in content.replace(f"import {imp}", ""):
                    unused.append(imp)
            
            return unused
            
        except Exception:
            return []
    
    async def _check_security_patterns(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """检查安全模式"""
        security_issues = []
        
        # 检查危险函数
        dangerous_functions = ['eval', 'exec', 'compile', '__import__']
        for func in dangerous_functions:
            if f"{func}(" in content:
                security_issues.append({
                    "type": "dangerous_function",
                    "function": func,
                    "message": f"使用了危险函数: {func}",
                    "severity": "high"
                })
        
        # 检查硬编码密码
        password_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'pwd\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']'
        ]
        
        for pattern in password_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                security_issues.append({
                    "type": "hardcoded_secret",
                    "message": "可能存在硬编码密码或密钥",
                    "severity": "high"
                })
        
        return security_issues
    
    async def _check_performance_bottlenecks(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """检查性能瓶颈"""
        bottlenecks = []
        
        # 检查循环中的数据库查询
        if re.search(r'for.*in.*:.*query', content, re.IGNORECASE):
            bottlenecks.append({
                "file": str(file_path.relative_to(self.project_root)),
                "type": "query_in_loop",
                "message": "循环中可能存在数据库查询",
                "severity": "medium"
            })
        
        # 检查大文件读取
        if 'file.read()' in content and 'with open' in content:
            bottlenecks.append({
                "file": str(file_path.relative_to(self.project_root)),
                "type": "large_file_read",
                "message": "可能存在大文件一次性读取",
                "severity": "medium"
            })
        
        return bottlenecks
    
    async def _generate_recommendations(self, scan_results: Dict[str, Any]) -> List[str]:
        """生成推荐建议"""
        recommendations = []
        
        # 基于发现的问题生成建议
        if len(scan_results["issues_found"]) > 10:
            recommendations.append("建议优先修复高严重性问题")
        
        if len(scan_results["duplicate_analysis"]) > 5:
            recommendations.append("发现较多重复代码，建议重构公共函数")
        
        if len(scan_results["unused_imports"]) > 3:
            recommendations.append("清理未使用的导入以提升代码质量")
        
        if len(scan_results["security_patterns"]) > 0:
            recommendations.append("存在安全问题，需要立即处理")
        
        if len(scan_results["performance_bottlenecks"]) > 0:
            recommendations.append("发现性能瓶颈，建议优化")
        
        return recommendations
    
    async def _run_test_suite(self, target: Optional[str] = None, test_type: str = "all", enable_coverage: bool = True) -> Dict[str, Any]:
        """运行测试套件（兼容原始版本功能）"""
        print(f"🧪 执行测试套件: {test_type if target else '全部'}")
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "target": target,
            "test_type": test_type,
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "coverage": {},
            "test_details": [],
            "execution_time": 0
        }
        
        try:
            # 构建pytest命令
            start_time = time.time()
            
            cmd = [sys.executable, "-m", "pytest"]
            
            # 添加目标路径
            if target:
                cmd.append(target)
            
            # 添加测试类型过滤器
            if test_type == "unit":
                cmd.extend(["-m", "unit"])
            elif test_type == "integration":
                cmd.extend(["-m", "integration"])
            elif test_type == "e2e":
                cmd.extend(["-m", "e2e"])
                # 对于E2E测试，激活Playwright MCP
                print("🌐 激活Playwright MCP进行端到端浏览器测试")
            
            # 添加覆盖率选项
            if enable_coverage:
                cmd.extend([
                    "--cov=.iflow/core",
                    "--cov-report=json",
                    "--cov-report=term-missing",
                    "--cov-report=html",
                    f"--cov-fail-under={self.config.coverage_threshold}"
                ])
            
            # 添加其他选项
            cmd.extend([
                "--tb=short",
                "-v",
                "--maxfail=5"
            ])
            
            # 执行测试
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            execution_time = time.time() - start_time
            test_results["execution_time"] = execution_time
            
            # 解析测试结果
            output = result.stdout
            error_output = result.stderr
            
            # 提取测试统计信息
            import re
            passed_match = re.search(r'(\d+)\s+passed', output)
            failed_match = re.search(r'(\d+)\s+failed', output)
            skipped_match = re.search(r'(\d+)\s+skipped', output)
            
            if passed_match:
                test_results["passed"] = int(passed_match.group(1))
            if failed_match:
                test_results["failed"] = int(failed_match.group(1))
            if skipped_match:
                test_results["skipped"] = int(skipped_match.group(1))
            
            # 读取覆盖率报告
            if enable_coverage:
                coverage_file = self.project_root / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                        test_results["coverage"] = coverage_data.get("totals", {})
                
                # 生成HTML覆盖率报告
                html_dir = self.project_root / "htmlcov"
                if html_dir.exists():
                    test_results["coverage_report"] = str(html_dir)
            
            test_results["test_details"] = output.split('\n')
            test_results["error_details"] = error_output.split('\n') if error_output else []
            
            # 智能测试失败分析
            if test_results["failed"] > 0:
                test_results["failure_analysis"] = await self._analyze_test_failures(output)
            
        except subprocess.TimeoutExpired:
            test_results["error"] = "测试执行超时"
        except Exception as e:
            test_results["error"] = f"测试执行错误: {e}"
        
        return test_results
    
    async def _analyze_test_failures(self, test_output: str) -> Dict[str, Any]:
        """分析测试失败原因（原始版本功能增强）"""
        failure_analysis = {
            "total_failures": 0,
            "failure_patterns": [],
            "recommendations": [],
            "common_errors": []
        }
        
        # 提取失败测试信息
        import re
        failure_pattern = r'FAILED\s+(.*?)::(.*?)\s*-\s*(.*)'
        failures = re.findall(failure_pattern, test_output)
        
        failure_analysis["total_failures"] = len(failures)
        
        # 分析失败模式
        for module, test, error in failures:
            failure_analysis["failure_patterns"].append({
                "module": module,
                "test": test,
                "error": error.strip()
            })
            
            # 生成针对性建议
            if "ImportError" in error:
                failure_analysis["recommendations"].append(
                    f"检查 {module} 的导入依赖"
                )
            elif "AssertionError" in error:
                failure_analysis["recommendations"].append(
                    f"检查 {test} 的断言逻辑"
                )
            elif "Timeout" in error:
                failure_analysis["recommendations"].append(
                    f"优化 {test} 的执行时间"
                )
        
        # 识别常见错误
        common_errors = re.findall(r'(ImportError|AttributeError|TypeError|ValueError|AssertionError)', test_output)
        failure_analysis["common_errors"] = list(set(common_errors))
        
        return failure_analysis
    
    async def _perform_security_scan(self) -> Dict[str, Any]:
        """执行安全扫描"""
        print("🛡️ 执行安全扫描...")
        
        security_results = {
            "timestamp": datetime.now().isoformat(),
            "scan_tool": "bandit",
            "total_issues": 0,
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0,
            "issues": []
        }
        
        try:
            # 运行bandit安全扫描
            cmd = [
                sys.executable, "-m", "bandit",
                "-r", ".iflow/core",
                "-f", "json",
                "-q"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                bandit_data = json.loads(result.stdout)
                security_results["issues"] = bandit_data.get("results", [])
                security_results["total_issues"] = len(security_results["issues"])
                
                # 按严重程度分类
                for issue in security_results["issues"]:
                    severity = issue.get("issue_severity", "LOW")
                    if severity == "HIGH":
                        security_results["high_severity"] += 1
                    elif severity == "MEDIUM":
                        security_results["medium_severity"] += 1
                    else:
                        security_results["low_severity"] += 1
            
        except Exception as e:
            security_results["error"] = f"安全扫描错误: {e}"
        
        return security_results
    
    async def _perform_performance_test(self) -> Dict[str, Any]:
        """执行性能测试"""
        print("⚡ 执行性能测试...")
        
        performance_results = {
            "timestamp": datetime.now().isoformat(),
            "memory_usage": {},
            "execution_times": {},
            "bottlenecks": []
        }
        
        try:
            # 测试内存使用
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            performance_results["memory_usage"] = {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent()
            }
            
            # 测试关键模块执行时间
            key_modules = [
                ".iflow/core/arq_engine_v16_1.py",
                ".iflow/core/hrrk_kernel_v3_enterprise.py",
                ".iflow/core/refrag_system_v6.py"
            ]
            
            for module in key_modules:
                module_path = self.project_root / module
                if module_path.exists():
                    start_time = time.time()
                    try:
                        # 简单的导入时间测试
                        spec = importlib.util.spec_from_file_location("test_module", module_path)
                        test_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(test_module)
                        exec_time = time.time() - start_time
                        performance_results["execution_times"][module] = exec_time
                    except Exception:
                        performance_results["execution_times"][module] = None
            
        except Exception as e:
            performance_results["error"] = f"性能测试错误: {e}"
        
        return performance_results
    
    async def _compare_project_structures(self) -> Dict[str, Any]:
        """比较项目结构变化"""
        if not self.project_structure_before or not self.project_structure_after:
            return {"status": "insufficient_data"}
        
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "files_added": [],
            "files_removed": [],
            "files_modified": [],
            "structure_changes": {},
            "complexity_changes": {},
            "dependency_changes": {}
        }
        
        before_files = set(self.project_structure_before.file_details.keys())
        after_files = set(self.project_structure_after.file_details.keys())
        
        # 找出新增文件
        comparison["files_added"] = list(after_files - before_files)
        
        # 找出删除文件
        comparison["files_removed"] = list(before_files - after_files)
        
        # 找出修改文件
        common_files = before_files & after_files
        for file_path in common_files:
            before_info = self.project_structure_before.file_details[file_path]
            after_info = self.project_structure_after.file_details[file_path]
            
            if before_info.modified_time != after_info.modified_time:
                comparison["files_modified"].append(file_path)
        
        return comparison
    
    async def _generate_optimization_report(self, test_results: Dict, security_results: Dict, 
                                         performance_results: Dict, scan_results: Dict, 
                                         structure_comparison: Dict) -> Dict[str, Any]:
        """生成优化报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "executive_summary": {},
            "test_analysis": {},
            "security_analysis": {},
            "performance_analysis": {},
            "code_quality_analysis": {},
            "structure_analysis": {},
            "recommendations": [],
            "action_items": []
        }
        
        # 执行摘要
        report["executive_summary"] = {
            "overall_health": "good" if test_results.get("passed", 0) > test_results.get("failed", 0) else "needs_attention",
            "critical_issues": len(security_results.get("high_severity", 0)),
            "test_coverage": test_results.get("coverage", {}).get("percent_covered", 0),
            "total_recommendations": len(scan_results.get("recommendations", []))
        }
        
        # 测试分析
        report["test_analysis"] = {
            "total_tests": test_results.get("total_tests", 0),
            "pass_rate": test_results.get("passed", 0) / max(test_results.get("total_tests", 1), 1),
            "coverage_score": test_results.get("coverage", {}).get("percent_covered", 0),
            "execution_time": test_results.get("execution_time", 0)
        }
        
        # 安全分析
        report["security_analysis"] = {
            "total_security_issues": security_results.get("total_issues", 0),
            "high_risk_issues": security_results.get("high_severity", 0),
            "medium_risk_issues": security_results.get("medium_severity", 0),
            "low_risk_issues": security_results.get("low_severity", 0)
        }
        
        # 性能分析
        report["performance_analysis"] = {
            "memory_usage_mb": performance_results.get("memory_usage", {}).get("rss", 0) / 1024 / 1024,
            "slow_modules": [k for k, v in performance_results.get("execution_times", {}).items() if v and v > 1.0]
        }
        
        # 代码质量分析
        report["code_quality_analysis"] = {
            "total_issues": len(scan_results.get("issues_found", [])),
            "duplicate_code_blocks": len(scan_results.get("duplicate_analysis", {})),
            "unused_imports": len(scan_results.get("unused_imports", {}))
        }
        
        # 结构分析
        report["structure_analysis"] = {
            "files_added": len(structure_comparison.get("files_added", [])),
            "files_removed": len(structure_comparison.get("files_removed", [])),
            "files_modified": len(structure_comparison.get("files_modified", []))
        }
        
        # 生成推荐建议
        report["recommendations"] = scan_results.get("recommendations", [])
        
        # 生成行动项
        if security_results.get("high_severity", 0) > 0:
            report["action_items"].append("立即处理高严重性安全问题")
        
        if test_results.get("coverage", {}).get("percent_covered", 0) < self.config.coverage_threshold:
            report["action_items"].append(f"提升测试覆盖率至{self.config.coverage_threshold}%以上")
        
        return report
    
    async def _interactive_analysis(self):
        """交互式分析"""
        print("\n🎯 进入交互式分析模式")
        print("=" * 50)
        
        while True:
            print("\n可选操作:")
            print("1. 查看详细测试结果")
            print("2. 查看安全扫描报告")
            print("3. 查看性能分析报告")
            print("4. 查看代码质量报告")
            print("5. 查看项目结构变化")
            print("6. 查看优化建议")
            print("7. 导出完整报告")
            print("0. 退出交互模式")
            
            choice = input("\n请选择操作 (0-7): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                await self._show_test_details()
            elif choice == "2":
                await self._show_security_details()
            elif choice == "3":
                await self._show_performance_details()
            elif choice == "4":
                await self._show_code_quality_details()
            elif choice == "5":
                await self._show_structure_details()
            elif choice == "6":
                await self._show_recommendations()
            elif choice == "7":
                await self._export_report()
            else:
                print("❌ 无效选择，请重试")
    
    async def _show_test_details(self):
        """显示测试详情"""
        print("\n📊 测试结果详情")
        print("-" * 40)
        
        if self.test_results:
            for key, value in self.test_results.items():
                print(f"{key}: {value}")
        else:
            print("暂无测试结果")
    
    async def _show_security_details(self):
        """显示安全详情"""
        print("\n🛡️ 安全扫描详情")
        print("-" * 40)
        # 实现安全详情显示逻辑
    
    async def _show_performance_details(self):
        """显示性能详情"""
        print("\n⚡ 性能分析详情")
        print("-" * 40)
        # 实现性能详情显示逻辑
    
    async def _show_code_quality_details(self):
        """显示代码质量详情"""
        print("\n📋 代码质量详情")
        print("-" * 40)
        # 实现代码质量详情显示逻辑
    
    async def _show_structure_details(self):
        """显示结构详情"""
        print("\n🏗️ 项目结构详情")
        print("-" * 40)
        # 实现结构详情显示逻辑
    
    async def _show_recommendations(self):
        """显示推荐建议"""
        print("\n💡 优化建议")
        print("-" * 40)
        
        if self.optimization_report.get("recommendations"):
            for i, rec in enumerate(self.optimization_report["recommendations"], 1):
                print(f"{i}. {rec}")
        else:
            print("暂无推荐建议")
    
    async def _export_report(self):
        """导出报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.project_root / f"enhanced_test_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.optimization_report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 报告已导出到: {report_file}")
    
    async def _generate_final_report(self) -> Dict[str, Any]:
        """生成最终报告"""
        final_report = {
            "timestamp": datetime.now().isoformat(),
            "test_configuration": asdict(self.config),
            "project_structure_before": asdict(self.project_structure_before) if self.project_structure_before else None,
            "project_structure_after": asdict(self.project_structure_after) if self.project_structure_after else None,
            "optimization_report": self.optimization_report,
            "ai_context_file": str(self.project_root / ".iflow" / "temp_docs" / "ai_context.json")
        }
        
        # 保存最终报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.project_root / f"enhanced_sc_test_final_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        
        print(f"📋 最终报告已保存到: {report_file}")
        
        return final_report
    
    async def _get_basic_structure(self) -> Dict[str, Any]:
        """获取基本项目结构"""
        structure = {
            "root_directories": [],
            "python_files": [],
            "config_files": [],
            "test_files": []
        }
        
        for item in self.project_root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                structure["root_directories"].append(item.name)
            elif item.is_file():
                if item.suffix == '.py':
                    structure["python_files"].append(item.name)
                elif item.name in ['pyproject.toml', 'setup.py', 'requirements.txt']:
                    structure["config_files"].append(item.name)
                elif 'test' in item.name.lower():
                    structure["test_files"].append(item.name)
        
        return structure

# 主函数
async def main():
    """主函数（兼容原始版本命令行接口）"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="增强版 /sc:test 指令",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                    # 运行完整增强分析
  %(prog)s src/core --type unit --coverage     # 单元测试与覆盖率
  %(prog)s --type e2e                          # 端到端浏览器测试
  %(prog)s --watch --fix                       # 监视模式（开发中）
  %(prog)s --no-interactive                    # 非交互模式
        """
    )
    
    # 原始版本参数
    parser.add_argument(
        "target", 
        nargs="?", 
        help="测试目标路径（如 src/components）"
    )
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "e2e", "all"],
        default="all",
        help="测试类型: unit(单元), integration(集成), e2e(端到端), all(全部)"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        default=True,
        help="启用覆盖率分析（默认启用）"
    )
    parser.add_argument(
        "--no-coverage", 
        action="store_true",
        help="禁用覆盖率分析"
    )
    parser.add_argument(
        "--watch", 
        action="store_true",
        help="连续监视模式（开发中）"
    )
    parser.add_argument(
        "--fix", 
        action="store_true",
        help="自动简单失败修复（开发中）"
    )
    
    # 增强版本参数
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="非交互模式"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="项目根目录路径"
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=25.0,
        help="覆盖率阈值（默认25%%）"
    )
    parser.add_argument(
        "--no-deep-analysis",
        action="store_true",
        help="禁用深度分析扫描"
    )
    parser.add_argument(
        "--no-optimization-report",
        action="store_true",
        help="禁用优化报告生成"
    )
    parser.add_argument(
        "--no-structure-comparison",
        action="store_true",
        help="禁用结构对比分析"
    )
    
    args = parser.parse_args()
    
    # 处理覆盖率选项
    enable_coverage = args.coverage and not args.no_coverage
    
    # 创建配置
    config = TestConfiguration(
        project_root=args.project_root,
        test_types=[args.type] if args.type != "all" else ["unit", "integration", "e2e"],
        coverage_threshold=args.coverage_threshold,
        interactive_mode=not args.no_interactive,
        enable_deep_analysis=not args.no_deep_analysis,
        force_ai_awareness=True
    )
    
    # 运行增强版测试
    async def run_test():
        engine = EnhancedTestEngine(config)
        
        # 如果是原始版本模式（指定了target或type），运行兼容模式
        if args.target or args.type != "all" or args.watch or args.fix:
            print("🔄 运行兼容模式 - 原始版本功能")
            results = await engine.run_compatible_test(
                target=args.target,
                test_type=args.type,
                enable_coverage=enable_coverage,
                watch_mode=args.watch,
                auto_fix=args.fix
            )
        else:
            # 运行完整增强版分析
            results = await engine.run_comprehensive_test()
        
        # 输出结果摘要
        if "test_results" in results:
            tr = results["test_results"]
            print(f"\n📊 测试摘要:")
            print(f"通过: {tr.get('passed', 0)}")
            print(f"失败: {tr.get('failed', 0)}")
            print(f"跳过: {tr.get('skipped', 0)}")
            print(f"执行时间: {tr.get('execution_time', 0):.2f}秒")
            
            if tr.get('coverage'):
                coverage_pct = tr['coverage'].get('percent_covered', 0)
                print(f"覆盖率: {coverage_pct:.1f}%")
        
        return results
    
    # 运行测试
    results = await run_test()
    
    print("\n🎉 增强版 /sc:test 执行完成！")
    print(f"📊 测试通过率: {results['optimization_report']['test_analysis']['pass_rate']:.2%}")
    print(f"🛡️ 安全问题: {results['optimization_report']['security_analysis']['total_security_issues']}个")
    print(f"📈 代码质量评分: {results['optimization_report']['code_quality_analysis']['total_issues']}个问题")

if __name__ == "__main__":
    asyncio.run(main())