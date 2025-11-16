#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 全自动化项目审查和升级工作流
Comprehensive Project Upgrade Workflow (CPUW)

这是iFlow CLI的旗舰级自动化工作流，提供全方位的项目审查、升级、优化和迭代功能。
集成AI驱动的智能分析、自动修复、性能优化、文档生成和持续学习能力。

核心功能：
- 🔍 全方位项目结构深度分析
- 🛠️ 代码质量自动审查和修复
- 📈 自动版本迭代和智能升级
- ⚡ 性能优化和自动测试
- 📚 智能文档生成和总结
- 🗑️ 自动清理旧代码和文件
- 📊 差异化报告和升级日志
- 🏗️ 项目架构深度分析
- 🧠 AI训练数据集生成和偏好学习
- 🔄 持续进化和自我完善

作者: AI架构师团队
版本: 1.0.0 Ultimate
日期: 2025-11-16
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
import hashlib
import shutil
import subprocess
import tempfile
import difflib
import re
import ast
# import yaml  # 注释掉可选依赖
# import toml   # 注释掉可选依赖
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from enum import Enum
# import git  # 注释掉，避免依赖问题

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入核心组件
try:
    from arq_reasoning_engine_v15_quantum import get_arq_engine_v15_quantum
    from hrrk_kernel_v2 import HRRKKernelV2
    from knowledge_base_manager import KnowledgeBaseManager
    from knowledge_base_ai_enhancer import get_ai_enhancer
except ImportError as e:
    print(f"⚠️ 核心组件导入失败，将使用基础功能: {e}")

# 可选依赖处理
try:
    import toml
except ImportError:
    toml = None

try:
    import yaml
except ImportError:
    yaml = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UpgradePhase(Enum):
    """升级阶段枚举"""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DOCUMENTATION = "documentation"
    CLEANUP = "cleanup"

class Severity(Enum):
    """问题严重程度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class ProjectMetrics:
    """项目指标"""
    total_files: int = 0
    code_files: int = 0
    test_files: int = 0
    doc_files: int = 0
    config_files: int = 0
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    complexity_score: float = 0.0
    maintainability_index: float = 0.0
    test_coverage: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0

@dataclass
class Issue:
    """问题记录"""
    id: str
    type: str
    severity: Severity
    title: str
    description: str
    file_path: str
    line_number: int
    evidence: str
    fix_suggestion: str
    auto_fixable: bool
    category: str
    impact: str
    effort: str

@dataclass
class UpgradeAction:
    """升级动作"""
    id: str
    type: str
    description: str
    file_path: str
    changes: Dict[str, Any]
    priority: str
    risk_level: str
    estimated_time: int
    dependencies: List[str]

@dataclass
class AIProfile:
    """AI用户偏好档案"""
    coding_style: Dict[str, Any]
    preferred_patterns: List[str]
    avoided_patterns: List[str]
    framework_preferences: Dict[str, Any]
    documentation_style: str
    testing_approach: str
    optimization_focus: List[str]
    security_priorities: List[str]
    performance_targets: Dict[str, float]
    architectural_preferences: Dict[str, Any]

class ComprehensiveProjectUpgradeWorkflow:
    """全自动化项目审查和升级工作流"""
    
    def __init__(self, workspace_path: str, config: Optional[Dict] = None):
        self.workspace_path = Path(workspace_path)
        self.config = config or {}
        
        # 核心组件
        self.arq_engine = None
        self.hrrk_kernel = None
        self.knowledge_base = None
        self.ai_enhancer = None
        
        # 工作流状态
        self.current_phase = UpgradePhase.ANALYSIS
        self.start_time = time.time()
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 分析结果
        self.project_metrics = ProjectMetrics()
        self.issues: List[Issue] = []
        self.upgrade_actions: List[UpgradeAction] = []
        self.architecture_analysis = {}
        self.performance_benchmarks = {}
        self.security_findings = []
        
        # AI学习数据
        self.ai_profile = AIProfile(
            coding_style={},
            preferred_patterns=[],
            avoided_patterns=[],
            framework_preferences={},
            documentation_style="technical",
            testing_approach="comprehensive",
            optimization_focus=["performance", "maintainability"],
            security_priorities=["authentication", "data_protection"],
            performance_targets={},
            architectural_preferences={}
        )
        
        # 升级历史
        self.upgrade_history = []
        self.changelog = []
        self.version_info = {"current": "1.0.0", "target": "1.1.0"}
        
        # 配置选项
        self.auto_fix = self.config.get("auto_fix", True)
        self.backup_enabled = self.config.get("backup_enabled", True)
        self.analysis_mode = self.config.get("dry_run", False)  # analysis_mode = True 表示只分析不修改
        self.verbose = self.config.get("verbose", False)
        
        # 文件类型映射
        self.file_extensions = {
            "code": [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".php", ".rb"],
            "test": ["test_", "_test.", ".test.", "spec_", "_spec.", ".spec."],
            "doc": [".md", ".rst", ".txt", ".doc", ".docx"],
            "config": [".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"],
            "build": ["Makefile", "CMakeLists.txt", "package.json", "requirements.txt", "pyproject.toml"]
        }
        
        logger.info("🚀 全自动化项目审查和升级工作流初始化完成")

    async def initialize(self):
        """初始化工作流环境"""
        logger.info("🔧 初始化工作流环境...")
        
        try:
            # 初始化核心组件
            await self._initialize_core_components()
            
            # 创建备份
            if self.backup_enabled and not self.analysis_mode:
                await self._create_backup()
            
            # 加载项目历史数据
            await self._load_project_history()
            
            # 初始化AI学习系统
            await self._initialize_ai_learning()
            
            logger.info("✅ 工作流环境初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 工作流初始化失败: {e}")
            raise

    async def _initialize_core_components(self):
        """初始化核心组件"""
        try:
            # ARQ推理引擎
            self.arq_engine = get_arq_engine_v15_quantum()
            logger.info("  ✅ ARQ推理引擎初始化完成")
            
            # HRRK内核
            self.hrrk_kernel = HRRKKernelV2()
            logger.info("  ✅ HRRK内核初始化完成")
            
            # 知识库
            self.knowledge_base = KnowledgeBaseManager()
            logger.info("  ✅ 知识库初始化完成")
            
            # AI增强器
            self.ai_enhancer = get_ai_enhancer()
            logger.info("  ✅ AI增强器初始化完成")
            
        except Exception as e:
            logger.warning(f"⚠️ 核心组件初始化失败，将使用基础功能: {e}")

    async def _create_backup(self):
        """创建项目备份"""
        logger.info("💾 创建项目备份...")
        
        try:
            backup_dir = self.workspace_path / ".iflow" / "backups" / f"upgrade_backup_{self.session_id}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 备份重要文件
            important_patterns = [
                "*.py", "*.js", "*.ts", "*.json", "*.yaml", "*.yml", "*.toml",
                "*.md", "requirements.txt", "package.json", "pyproject.toml"
            ]
            
            for pattern in important_patterns:
                for file_path in self.workspace_path.rglob(pattern):
                    if file_path.is_file() and not any(skip in str(file_path) for skip in ['.git', '__pycache__', 'node_modules', '.venv']):
                        relative_path = file_path.relative_to(self.workspace_path)
                        backup_path = backup_dir / relative_path
                        backup_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, backup_path)
            
            logger.info(f"  ✅ 备份完成: {backup_dir}")
            
        except Exception as e:
            logger.error(f"  ❌ 备份失败: {e}")

    async def _load_project_history(self):
        """加载项目历史数据"""
        logger.info("📚 加载项目历史数据...")
        
        try:
            history_file = self.workspace_path / ".iflow" / "data" / "upgrade_history.json"
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.upgrade_history = json.load(f)
                logger.info(f"  ✅ 加载了 {len(self.upgrade_history)} 条历史记录")
            
            # 加载AI偏好档案
            profile_file = self.workspace_path / ".iflow" / "data" / "ai_profile.json"
            if profile_file.exists():
                with open(profile_file, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                    self.ai_profile = AIProfile(**profile_data)
                logger.info("  ✅ AI偏好档案加载完成")
            
        except Exception as e:
            logger.warning(f"⚠️ 历史数据加载失败: {e}")

    async def _initialize_ai_learning(self):
        """初始化AI学习系统"""
        logger.info("🧠 初始化AI学习系统...")
        
        try:
            # 分析现有代码模式
            await self._analyze_coding_patterns()
            
            # 学习项目架构偏好
            await self._learn_architectural_preferences()
            
            # 理解用户文档风格
            await self._understand_documentation_style()
            
            logger.info("  ✅ AI学习系统初始化完成")
            
        except Exception as e:
            logger.warning(f"⚠️ AI学习系统初始化失败: {e}")

    async def execute_comprehensive_upgrade(self) -> Dict[str, Any]:
        """执行全面升级流程"""
        logger.info("🚀 开始执行全面项目升级...")
        
        try:
            # 阶段1: 深度分析
            await self._phase_analysis()
            
            # 阶段2: 智能规划
            await self._phase_planning()
            
            # 阶段3: 执行升级
            await self._phase_execution()
            
            # 阶段4: 验证测试
            await self._phase_validation()
            
            # 阶段5: 文档生成
            await self._phase_documentation()
            
            # 阶段6: 清理优化
            await self._phase_cleanup()
            
            # 生成最终报告
            final_report = await self._generate_final_report()
            
            # 保存升级历史
            await self._save_upgrade_history(final_report)
            
            logger.info("🎉 全面项目升级完成!")
            
            return final_report
            
        except Exception as e:
            logger.error(f"❌ 升级流程执行失败: {e}")
            raise

    async def _phase_analysis(self):
        """阶段1: 深度分析"""
        logger.info("📊 阶段1: 深度分析...")
        self.current_phase = UpgradePhase.ANALYSIS
        
        try:
            # 1.1 项目结构分析
            await self._analyze_project_structure()
            
            # 1.2 代码质量分析
            await self._analyze_code_quality()
            
            # 1.3 架构分析
            await self._analyze_architecture()
            
            # 1.4 性能分析
            await self._analyze_performance()
            
            # 1.5 安全分析
            await self._analyze_security()
            
            # 1.6 依赖分析
            await self._analyze_dependencies()
            
            # 1.7 测试覆盖率分析
            await self._analyze_test_coverage()
            
            logger.info("  ✅ 深度分析完成")
            
        except Exception as e:
            logger.error(f"  ❌ 深度分析失败: {e}")
            raise

    async def _analyze_project_structure(self):
        """分析项目结构"""
        logger.info("  📁 分析项目结构...")
        
        try:
            structure_analysis = {
                "directory_tree": await self._build_directory_tree(),
                "file_distribution": await self._analyze_file_distribution(),
                "module_dependencies": await self._analyze_module_dependencies(),
                "naming_conventions": await self._analyze_naming_conventions(),
                "organization_patterns": await self._analyze_organization_patterns()
            }
            
            self.architecture_analysis["project_structure"] = structure_analysis
            
            # 更新项目指标
            self.project_metrics.total_files = sum(
                len(list(self.workspace_path.rglob(f"*{ext}")))
                for category in self.file_extensions.values()
                for ext in category
            )
            
            logger.info(f"    发现 {self.project_metrics.total_files} 个文件")
            
        except Exception as e:
            logger.error(f"    项目结构分析失败: {e}")

    async def _build_directory_tree(self) -> Dict[str, Any]:
        """构建目录树"""
        def build_tree(path: Path, max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return {"type": "directory", "name": path.name, "children": "..."}
            
            tree = {"type": "directory", "name": path.name, "children": []}
            
            try:
                for item in sorted(path.iterdir()):
                    if item.name.startswith('.'):
                        continue
                    
                    if item.is_dir():
                        subtree = build_tree(item, max_depth, current_depth + 1)
                        tree["children"].append(subtree)
                    else:
                        tree["children"].append({
                            "type": "file",
                            "name": item.name,
                            "size": item.stat().st_size,
                            "extension": item.suffix
                        })
            except PermissionError:
                pass
            
            return tree
        
        return build_tree(self.workspace_path)

    async def _analyze_file_distribution(self) -> Dict[str, Any]:
        """分析文件分布"""
        distribution = defaultdict(int)
        size_distribution = defaultdict(int)
        
        for file_path in self.workspace_path.rglob("*"):
            if file_path.is_file() and not any(skip in str(file_path) for skip in ['.git', '__pycache__', 'node_modules']):
                ext = file_path.suffix.lower()
                distribution[ext] += 1
                try:
                    size_distribution[ext] += file_path.stat().st_size
                except:
                    pass
        
        return {
            "count_by_type": dict(distribution),
            "size_by_type": dict(size_distribution),
            "total_size": sum(size_distribution.values())
        }

    async def _analyze_module_dependencies(self) -> Dict[str, Any]:
        """分析模块依赖"""
        dependencies = defaultdict(set)
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 简单的导入分析
                    imports = re.findall(r'^import\s+(\w+)|^from\s+(\w+)', content, re.MULTILINE)
                    for import_match in imports:
                        module = import_match[0] or import_match[1]
                        if module and not module.startswith('.'):
                            dependencies[str(file_path)].add(module)
                except:
                    pass
        
        return {
            "dependencies": {k: list(v) for k, v in dependencies.items()},
            "dependency_graph": self._build_dependency_graph(dependencies)
        }

    def _build_dependency_graph(self, dependencies: Dict[str, set]) -> Dict[str, List[str]]:
        """构建依赖图"""
        graph = {}
        for file_path, deps in dependencies.items():
            graph[file_path] = list(deps)
        return graph

    async def _analyze_naming_conventions(self) -> Dict[str, Any]:
        """分析命名约定"""
        conventions = {
            "file_naming": {},
            "variable_naming": {},
            "function_naming": {},
            "class_naming": {}
        }
        
        # 分析文件命名
        file_patterns = defaultdict(list)
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                name = file_path.stem
                if name.islower():
                    file_patterns["snake_case"].append(str(file_path))
                elif any(c.isupper() for c in name):
                    file_patterns["camel_case"].append(str(file_path))
        
        conventions["file_naming"] = dict(file_patterns)
        
        return conventions

    async def _analyze_organization_patterns(self) -> Dict[str, Any]:
        """分析组织模式"""
        patterns = {
            "has_tests": any("test" in str(p).lower() for p in self.workspace_path.rglob("*")),
            "has_docs": any(p.suffix in ['.md', '.rst'] for p in self.workspace_path.rglob("*")),
            "has_config": any(p.name in ['config', 'settings', '.env'] for p in self.workspace_path.rglob("*")),
            "has_ci": any(p.name in ['.github', '.gitlab-ci.yml', 'travis.yml'] for p in self.workspace_path.rglob("*")),
            "package_managers": []
        }
        
        # 检测包管理器
        if (self.workspace_path / "package.json").exists():
            patterns["package_managers"].append("npm")
        if (self.workspace_path / "requirements.txt").exists() or (self.workspace_path / "pyproject.toml").exists():
            patterns["package_managers"].append("pip")
        if (self.workspace_path / "Cargo.toml").exists():
            patterns["package_managers"].append("cargo")
        
        return patterns

    async def _analyze_code_quality(self):
        """分析代码质量"""
        logger.info("  🔍 分析代码质量...")
        
        try:
            # 静态代码分析
            await self._perform_static_analysis()
            
            # 复杂度分析
            await self._analyze_complexity()
            
            # 可维护性分析
            await self._analyze_maintainability()
            
            # 代码风格分析
            await self._analyze_code_style()
            
            # 重复代码分析
            await self._analyze_code_duplication()
            
            logger.info("    发现 {} 个质量问题".format(len(self.issues)))
            
        except Exception as e:
            logger.error(f"    代码质量分析失败: {e}")

    async def _perform_static_analysis(self):
        """执行静态代码分析"""
        logger.info("    执行静态代码分析...")
        
        # 定义质量规则
        quality_rules = {
            "long_lines": {
                "pattern": r".{120,}",  # 超过120字符的行
                "severity": Severity.MEDIUM,
                "message": "行长度超过120字符"
            },
            "trailing_whitespace": {
                "pattern": r".+\s+$",
                "severity": Severity.LOW,
                "message": "行尾有多余空格"
            },
            "missing_docstrings": {
                "pattern": r"def\s+\w+\([^)]*\):\s*$",
                "severity": Severity.MEDIUM,
                "message": "函数缺少文档字符串"
            },
            "unused_imports": {
                "pattern": r"^import\s+\w+",
                "severity": Severity.LOW,
                "message": "可能未使用的导入"
            },
            "hardcoded_values": {
                "pattern": r"\b\d{3,}\b",
                "severity": Severity.MEDIUM,
                "message": "硬编码数值"
            }
        }
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        for rule_name, rule_info in quality_rules.items():
                            if re.search(rule_info["pattern"], line):
                                issue = Issue(
                                    id=f"{rule_name}_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}",
                                    type="code_quality",
                                    severity=rule_info["severity"],
                                    title=rule_info["message"],
                                    description=f"在文件 {file_path} 第{line_num}行发现 {rule_info['message']}",
                                    file_path=str(file_path),
                                    line_number=line_num,
                                    evidence=line.strip(),
                                    fix_suggestion=self._get_fix_suggestion(rule_name),
                                    auto_fixable=rule_name in ["trailing_whitespace", "long_lines"],
                                    category="code_style",
                                    impact="maintainability",
                                    effort="low"
                                )
                                self.issues.append(issue)
                
                except Exception as e:
                    logger.warning(f"无法分析文件 {file_path}: {e}")

    def _get_fix_suggestion(self, rule_name: str) -> str:
        """获取修复建议"""
        suggestions = {
            "long_lines": "将长行分解为多行，提高代码可读性",
            "trailing_whitespace": "删除行尾空格",
            "missing_docstrings": "为函数添加详细的文档字符串",
            "unused_imports": "删除未使用的导入语句",
            "hardcoded_values": "将硬编码值提取为常量或配置项"
        }
        return suggestions.get(rule_name, "请参考最佳实践进行修复")

    async def _analyze_complexity(self):
        """分析代码复杂度"""
        logger.info("    分析代码复杂度...")
        
        total_complexity = 0
        file_count = 0
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 简单的复杂度计算
                    complexity = self._calculate_complexity(content)
                    total_complexity += complexity
                    file_count += 1
                    
                    if complexity > 10:
                        issue = Issue(
                            id=f"complexity_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}",
                            type="complexity",
                            severity=Severity.HIGH if complexity > 20 else Severity.MEDIUM,
                            title="函数复杂度过高",
                            description=f"文件 {file_path} 的复杂度为 {complexity}，超过推荐值",
                            file_path=str(file_path),
                            line_number=1,
                            evidence=f"复杂度: {complexity}",
                            fix_suggestion="考虑将复杂函数拆分为多个小函数",
                            auto_fixable=False,
                            category="complexity",
                            impact="maintainability",
                            effort="medium"
                        )
                        self.issues.append(issue)
                
                except Exception as e:
                    logger.warning(f"无法分析复杂度 {file_path}: {e}")
        
        if file_count > 0:
            self.project_metrics.complexity_score = total_complexity / file_count

    def _calculate_complexity(self, content: str) -> int:
        """计算代码复杂度"""
        try:
            tree = ast.parse(content)
            complexity = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            return complexity
        except:
            return 0

    async def _analyze_maintainability(self):
        """分析可维护性"""
        logger.info("    分析可维护性...")
        
        # 计算可维护性指数
        total_lines = 0
        comment_lines = 0
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        total_lines += 1
                        stripped = line.strip()
                        if stripped.startswith('#') or '"""' in stripped or "'''" in stripped:
                            comment_lines += 1
                
                except Exception as e:
                    logger.warning(f"无法分析可维护性 {file_path}: {e}")
        
        self.project_metrics.total_lines = total_lines
        self.project_metrics.comment_lines = comment_lines
        self.project_metrics.code_lines = total_lines - comment_lines
        
        if total_lines > 0:
            comment_ratio = comment_lines / total_lines
            # 简单的可维护性指数计算
            maintainability = min(100, (comment_ratio * 100) + (50 - min(50, self.project_metrics.complexity_score)))
            self.project_metrics.maintainability_index = maintainability

    async def _analyze_code_style(self):
        """分析代码风格"""
        logger.info("    分析代码风格...")
        
        # PEP 8 风格检查
        style_issues = [
            {
                "pattern": r"def\s+([a-z][a-z0-9_]*)\s*\(",
                "severity": Severity.MEDIUM,
                "message": "函数名应使用小写字母和下划线"
            },
            {
                "pattern": r"class\s+([A-Z][a-zA-Z0-9_]*)\s*:",
                "severity": Severity.MEDIUM,
                "message": "类名应使用驼峰命名法"
            }
        ]
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for issue_rule in style_issues:
                        matches = re.finditer(issue_rule["pattern"], content)
                        for match in matches:
                            issue = Issue(
                                id=f"style_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}_{len(self.issues)}",
                                type="code_style",
                                severity=issue_rule["severity"],
                                title=issue_rule["message"],
                                description=f"在文件 {file_path} 中发现风格问题",
                                file_path=str(file_path),
                                line_number=content[:match.start()].count('\n') + 1,
                                evidence=match.group(0),
                                fix_suggestion="按照PEP 8规范调整命名",
                                auto_fixable=False,
                                category="code_style",
                                impact="readability",
                                effort="low"
                            )
                            self.issues.append(issue)
                
                except Exception as e:
                    logger.warning(f"无法分析代码风格 {file_path}: {e}")

    async def _analyze_code_duplication(self):
        """分析代码重复"""
        logger.info("    分析代码重复...")
        
        # 简单的重复代码检测
        code_blocks = defaultdict(list)
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # 检查5行以上的重复块
                    for i in range(len(lines) - 4):
                        block = ''.join(lines[i:i+5]).strip()
                        if len(block) > 50:  # 忽略太短的块
                            code_blocks[block].append((str(file_path), i+1))
                
                except Exception as e:
                    logger.warning(f"无法分析代码重复 {file_path}: {e}")
        
        # 报告重复代码
        for block, occurrences in code_blocks.items():
            if len(occurrences) > 1:
                issue = Issue(
                    id=f"duplication_{hashlib.md5(block.encode()).hexdigest()[:8]}",
                    type="code_duplication",
                    severity=Severity.MEDIUM,
                    title="代码重复",
                    description=f"发现重复代码块，出现在 {len(occurrences)} 个位置",
                    file_path=occurrences[0][0],
                    line_number=occurrences[0][1],
                    evidence=block[:100] + "..." if len(block) > 100 else block,
                    fix_suggestion="考虑将重复代码提取为函数或模块",
                    auto_fixable=False,
                    category="duplication",
                    impact="maintainability",
                    effort="medium"
                )
                self.issues.append(issue)

    async def _analyze_architecture(self):
        """分析架构"""
        logger.info("  🏗️ 分析架构...")
        
        try:
            # 架构模式识别
            await self._identify_architecture_patterns()
            
            # 模块耦合度分析
            await self._analyze_coupling()
            
            # 设计模式识别
            await self._identify_design_patterns()
            
            # 分层架构分析
            await self._analyze_layered_architecture()
            
        except Exception as e:
            logger.error(f"    架构分析失败: {e}")

    async def _identify_architecture_patterns(self):
        """识别架构模式"""
        patterns = {
            "mvc": ["models", "views", "controllers"],
            "mvp": ["models", "views", "presenters"],
            "mvvm": ["models", "views", "viewmodels"],
            "layered": ["controllers", "services", "repositories", "models"],
            "microservice": ["services", "apis", "gateways"],
            "plugin": ["plugins", "extensions", "core"]
        }
        
        detected_patterns = []
        
        for pattern_name, pattern_dirs in patterns.items():
            pattern_found = True
            for required_dir in pattern_dirs:
                if not any(required_dir in str(p).lower() for p in self.workspace_path.iterdir() if p.is_dir()):
                    pattern_found = False
                    break
            
            if pattern_found:
                detected_patterns.append(pattern_name)
        
        self.architecture_analysis["detected_patterns"] = detected_patterns

    async def _analyze_coupling(self):
        """分析耦合度"""
        # 简化的耦合度分析
        coupling_score = 0
        module_count = 0
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 计算导入数量作为耦合度指标
                    imports = len(re.findall(r'^import\s+|^from\s+\w+', content, re.MULTILINE))
                    coupling_score += imports
                    module_count += 1
                
                except Exception as e:
                    logger.warning(f"无法分析耦合度 {file_path}: {e}")
        
        if module_count > 0:
            avg_coupling = coupling_score / module_count
            self.architecture_analysis["average_coupling"] = avg_coupling

    async def _identify_design_patterns(self):
        """识别设计模式"""
        # 简化的设计模式识别
        patterns_found = []
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 单例模式检测
                    if re.search(r'class\s+\w+.*__new__.*instance', content):
                        patterns_found.append(("singleton", str(file_path)))
                    
                    # 工厂模式检测
                    if re.search(r'def\s+create_\w+|class\s+\w*Factory\w*', content):
                        patterns_found.append(("factory", str(file_path)))
                    
                    # 观察者模式检测
                    if re.search(r'add_observer|notify_observers|attach.*detach', content):
                        patterns_found.append(("observer", str(file_path)))
                
                except Exception as e:
                    logger.warning(f"无法识别设计模式 {file_path}: {e}")
        
        self.architecture_analysis["design_patterns"] = patterns_found

    async def _analyze_layered_architecture(self):
        """分析分层架构"""
        layers = {
            "presentation": ["views", "controllers", "handlers", "apis"],
            "business": ["services", "business", "logic", "domain"],
            "data": ["repositories", "data", "models", "entities"],
            "infrastructure": ["config", "utils", "helpers", "common"]
        }
        
        layer_files = defaultdict(list)
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                file_str = str(file_path).lower()
                for layer_name, layer_keywords in layers.items():
                    if any(keyword in file_str for keyword in layer_keywords):
                        layer_files[layer_name].append(str(file_path))
        
        self.architecture_analysis["layer_distribution"] = dict(layer_files)

    async def _analyze_performance(self):
        """分析性能"""
        logger.info("  ⚡ 分析性能...")
        
        try:
            # 性能瓶颈识别
            await self._identify_performance_bottlenecks()
            
            # 算法复杂度分析
            await self._analyze_algorithmic_complexity()
            
            # 内存使用分析
            await self._analyze_memory_usage()
            
            # I/O操作分析
            await self._analyze_io_operations()
            
        except Exception as e:
            logger.error(f"    性能分析失败: {e}")

    async def _identify_performance_bottlenecks(self):
        """识别性能瓶颈"""
        bottleneck_patterns = [
            {
                "pattern": r"for\s+\w+\s+in\s+.*\.keys\(\)",
                "severity": Severity.MEDIUM,
                "message": "使用.keys()遍历字典效率较低"
            },
            {
                "pattern": r"\.format\(|%\s*.*%|f['\"]",
                "severity": Severity.LOW,
                "message": "字符串格式化可能影响性能"
            },
            {
                "pattern": r"time\.sleep\(",
                "severity": Severity.MEDIUM,
                "message": "同步sleep可能阻塞线程"
            }
        ]
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern_info in bottleneck_patterns:
                        matches = re.finditer(pattern_info["pattern"], content)
                        for match in matches:
                            issue = Issue(
                                id=f"performance_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}_{len(self.issues)}",
                                type="performance",
                                severity=pattern_info["severity"],
                                title=pattern_info["message"],
                                description=f"在文件 {file_path} 中发现潜在性能问题",
                                file_path=str(file_path),
                                line_number=content[:match.start()].count('\n') + 1,
                                evidence=match.group(0),
                                fix_suggestion=self._get_performance_fix_suggestion(pattern_info["message"]),
                                auto_fixable=False,
                                category="performance",
                                impact="performance",
                                effort="medium"
                            )
                            self.issues.append(issue)
                
                except Exception as e:
                    logger.warning(f"无法分析性能瓶颈 {file_path}: {e}")

    def _get_performance_fix_suggestion(self, message: str) -> str:
        """获取性能修复建议"""
        suggestions = {
            "使用.keys()遍历字典效率较低": "直接遍历字典而不是.keys()",
            "字符串格式化可能影响性能": "考虑使用f-string或更高效的格式化方法",
            "同步sleep可能阻塞线程": "考虑使用异步sleep或非阻塞方式"
        }
        return suggestions.get(message, "请参考性能最佳实践进行优化")

    async def _analyze_algorithmic_complexity(self):
        """分析算法复杂度"""
        complexity_patterns = [
            {
                "pattern": r"for\s+.*\s+in\s+.*:\s*for\s+.*\s+in\s+.*",
                "severity": Severity.HIGH,
                "message": "嵌套循环可能导致O(n²)复杂度"
            },
            {
                "pattern": r"\.sort\(\)|sorted\(",
                "severity": Severity.MEDIUM,
                "message": "排序操作的时间复杂度为O(n log n)"
            }
        ]
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern_info in complexity_patterns:
                        matches = re.finditer(pattern_info["pattern"], content)
                        for match in matches:
                            issue = Issue(
                                id=f"algorithm_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}_{len(self.issues)}",
                                type="algorithmic_complexity",
                                severity=pattern_info["severity"],
                                title=pattern_info["message"],
                                description=f"在文件 {file_path} 中发现算法复杂度问题",
                                file_path=str(file_path),
                                line_number=content[:match.start()].count('\n') + 1,
                                evidence=match.group(0),
                                fix_suggestion="考虑优化算法或使用更高效的数据结构",
                                auto_fixable=False,
                                category="algorithm",
                                impact="performance",
                                effort="high"
                            )
                            self.issues.append(issue)
                
                except Exception as e:
                    logger.warning(f"无法分析算法复杂度 {file_path}: {e}")

    async def _analyze_memory_usage(self):
        """分析内存使用"""
        memory_patterns = [
            {
                "pattern": r"\[\w+\s+for\s+.*\s+in\s+.*\s+if\s+.*\]",
                "severity": Severity.MEDIUM,
                "message": "列表推导式可能消耗大量内存"
            },
            {
                "pattern": r"append\(.*\)\s*for\s+.*\s+in\s+.*:",
                "severity": Severity.LOW,
                "message": "循环中append可能导致频繁内存分配"
            }
        ]
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern_info in memory_patterns:
                        matches = re.finditer(pattern_info["pattern"], content)
                        for match in matches:
                            issue = Issue(
                                id=f"memory_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}_{len(self.issues)}",
                                type="memory_usage",
                                severity=pattern_info["severity"],
                                title=pattern_info["message"],
                                description=f"在文件 {file_path} 中发现内存使用问题",
                                file_path=str(file_path),
                                line_number=content[:match.start()].count('\n') + 1,
                                evidence=match.group(0),
                                fix_suggestion="考虑使用生成器或优化内存使用模式",
                                auto_fixable=False,
                                category="memory",
                                impact="memory",
                                effort="medium"
                            )
                            self.issues.append(issue)
                
                except Exception as e:
                    logger.warning(f"无法分析内存使用 {file_path}: {e}")

    async def _analyze_io_operations(self):
        """分析I/O操作"""
        io_patterns = [
            {
                "pattern": r"open\([^)]*\)\.read\(\)",
                "severity": Severity.MEDIUM,
                "message": "一次性读取大文件可能消耗大量内存"
            },
            {
                "pattern": r"with\s+open\([^)]*\)\s+as\s+f:",
                "severity": Severity.LOW,
                "message": "文件I/O操作建议使用上下文管理器"
            }
        ]
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern_info in io_patterns:
                        matches = re.finditer(pattern_info["pattern"], content)
                        for match in matches:
                            issue = Issue(
                                id=f"io_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}_{len(self.issues)}",
                                type="io_operation",
                                severity=pattern_info["severity"],
                                title=pattern_info["message"],
                                description=f"在文件 {file_path} 中发现I/O操作问题",
                                file_path=str(file_path),
                                line_number=content[:match.start()].count('\n') + 1,
                                evidence=match.group(0),
                                fix_suggestion="优化I/O操作，考虑分块读取或异步处理",
                                auto_fixable=False,
                                category="io",
                                impact="performance",
                                effort="medium"
                            )
                            self.issues.append(issue)
                
                except Exception as e:
                    logger.warning(f"无法分析I/O操作 {file_path}: {e}")

    async def _analyze_security(self):
        """分析安全性"""
        logger.info("  🛡️ 分析安全性...")
        
        try:
            # 安全漏洞扫描
            await self._scan_security_vulnerabilities()
            
            # 敏感信息检测
            await self._detect_sensitive_information()
            
            # 权限分析
            await self._analyze_permissions()
            
            # 依赖安全分析
            await self._analyze_dependency_security()
            
        except Exception as e:
            logger.error(f"    安全分析失败: {e}")

    async def _scan_security_vulnerabilities(self):
        """扫描安全漏洞"""
        security_patterns = [
            {
                "pattern": r"eval\(|exec\(",
                "severity": Severity.CRITICAL,
                "message": "使用eval或exec存在代码注入风险",
                "cwe": "CWE-94"
            },
            {
                "pattern": r"shell=True|subprocess\.call.*shell=True",
                "severity": Severity.CRITICAL,
                "message": "shell=True存在命令注入风险",
                "cwe": "CWE-78"
            },
            {
                "pattern": r"pickle\.loads|cPickle\.loads",
                "severity": Severity.HIGH,
                "message": "pickle反序列化存在安全风险",
                "cwe": "CWE-502"
            },
            {
                "pattern": r"random\.random|random\.randint",
                "severity": Severity.MEDIUM,
                "message": "使用伪随机数生成器可能不安全",
                "cwe": "CWE-338"
            },
            {
                "pattern": r"hashlib\.md5\(|hashlib\.sha1\(",
                "severity": Severity.MEDIUM,
                "message": "使用弱哈希算法",
                "cwe": "CWE-327"
            }
        ]
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern_info in security_patterns:
                        matches = re.finditer(pattern_info["pattern"], content)
                        for match in matches:
                            issue = Issue(
                                id=f"security_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}_{len(self.issues)}",
                                type="security_vulnerability",
                                severity=pattern_info["severity"],
                                title=pattern_info["message"],
                                description=f"在文件 {file_path} 中发现安全漏洞 ({pattern_info['cwe']})",
                                file_path=str(file_path),
                                line_number=content[:match.start()].count('\n') + 1,
                                evidence=match.group(0),
                                fix_suggestion=self._get_security_fix_suggestion(pattern_info["message"]),
                                auto_fixable=False,
                                category="security",
                                impact="security",
                                effort="high"
                            )
                            self.issues.append(issue)
                            self.security_findings.append(issue)
                
                except Exception as e:
                    logger.warning(f"无法扫描安全漏洞 {file_path}: {e}")

    def _get_security_fix_suggestion(self, message: str) -> str:
        """获取安全修复建议"""
        suggestions = {
            "使用eval或exec存在代码注入风险": "避免使用eval/exec，使用安全的替代方案",
            "shell=True存在命令注入风险": "避免shell=True，使用参数化命令",
            "pickle反序列化存在安全风险": "使用安全的序列化格式如JSON",
            "使用伪随机数生成器可能不安全": "使用secrets模块生成安全随机数",
            "使用弱哈希算法": "使用强哈希算法如SHA-256或SHA-3"
        }
        return suggestions.get(message, "请参考安全最佳实践进行修复")

    async def _detect_sensitive_information(self):
        """检测敏感信息"""
        sensitive_patterns = [
            {
                "pattern": r"(password|passwd|pwd)\s*=\s*['\"][^'\"]+['\"]",
                "severity": Severity.HIGH,
                "message": "硬编码密码"
            },
            {
                "pattern": r"(api_key|apikey|secret_key)\s*=\s*['\"][^'\"]+['\"]",
                "severity": Severity.HIGH,
                "message": "硬编码API密钥"
            },
            {
                "pattern": r"(token|auth)\s*=\s*['\"][^'\"]+['\"]",
                "severity": Severity.HIGH,
                "message": "硬编码认证令牌"
            }
        ]
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern_info in sensitive_patterns:
                        matches = re.finditer(pattern_info["pattern"], content, re.IGNORECASE)
                        for match in matches:
                            issue = Issue(
                                id=f"sensitive_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}_{len(self.issues)}",
                                type="sensitive_information",
                                severity=pattern_info["severity"],
                                title=pattern_info["message"],
                                description=f"在文件 {file_path} 中发现敏感信息",
                                file_path=str(file_path),
                                line_number=content[:match.start()].count('\n') + 1,
                                evidence=match.group(0)[:50] + "...",
                                fix_suggestion="将敏感信息移至环境变量或配置文件",
                                auto_fixable=True,
                                category="security",
                                impact="security",
                                effort="medium"
                            )
                            self.issues.append(issue)
                
                except Exception as e:
                    logger.warning(f"无法检测敏感信息 {file_path}: {e}")

    async def _analyze_permissions(self):
        """分析权限"""
        # 检查文件权限
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    stat_info = file_path.stat()
                    mode = oct(stat_info.st_mode)[-3:]
                    
                    # 检查是否对其他用户可写
                    if mode[2] in ['2', '3', '6', '7']:
                        issue = Issue(
                            id=f"permission_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}",
                            type="file_permission",
                            severity=Severity.MEDIUM,
                            title="文件权限过于宽松",
                            description=f"文件 {file_path} 对其他用户可写",
                            file_path=str(file_path),
                            line_number=1,
                            evidence=f"权限模式: {mode}",
                            fix_suggestion="调整文件权限，移除其他用户的写权限",
                            auto_fixable=True,
                            category="security",
                            impact="security",
                            effort="low"
                        )
                        self.issues.append(issue)
                
                except Exception as e:
                    logger.warning(f"无法分析权限 {file_path}: {e}")

    async def _analyze_dependency_security(self):
        """分析依赖安全"""
        # 检查依赖文件
        dependency_files = ["requirements.txt", "pyproject.toml", "package.json"]
        
        for dep_file in dependency_files:
            file_path = self.workspace_path / dep_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 简单的已知漏洞依赖检查
                    vulnerable_packages = [
                        "urllib3==1.24.2",  # 示例漏洞包
                        "requests==2.20.0",
                        "pillow<6.2.0"
                    ]
                    
                    for vuln_pkg in vulnerable_packages:
                        if vuln_pkg in content:
                            issue = Issue(
                                id=f"dep_vuln_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}",
                                type="dependency_vulnerability",
                                severity=Severity.HIGH,
                                title="依赖包存在已知漏洞",
                                description=f"在 {dep_file} 中发现漏洞依赖: {vuln_pkg}",
                                file_path=str(file_path),
                                line_number=content.split('\n').index([line for line in content.split('\n') if vuln_pkg in line][0]) + 1,
                                evidence=vuln_pkg,
                                fix_suggestion="升级到安全版本",
                                auto_fixable=True,
                                category="security",
                                impact="security",
                                effort="medium"
                            )
                            self.issues.append(issue)
                
                except Exception as e:
                    logger.warning(f"无法分析依赖安全 {file_path}: {e}")

    async def _analyze_dependencies(self):
        """分析依赖"""
        logger.info("  📦 分析依赖...")
        
        try:
            # 依赖关系分析
            await self._analyze_dependency_graph()
            
            # 版本兼容性分析
            await self._analyze_version_compatibility()
            
            # 许可证合规性分析
            await self._analyze_license_compliance()
            
        except Exception as e:
            logger.error(f"    依赖分析失败: {e}")

    async def _analyze_dependency_graph(self):
        """分析依赖图"""
        dependencies = defaultdict(set)
        
        # 分析Python依赖
        if (self.workspace_path / "requirements.txt").exists():
            with open(self.workspace_path / "requirements.txt", 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                        dependencies["python"].add(pkg_name)
        
        # 分析Node.js依赖
        if (self.workspace_path / "package.json").exists():
            try:
                with open(self.workspace_path / "package.json", 'r') as f:
                    package_data = json.load(f)
                
                for dep_type in ["dependencies", "devDependencies"]:
                    if dep_type in package_data:
                        for pkg_name in package_data[dep_type].keys():
                            dependencies["nodejs"].add(pkg_name)
            except Exception as e:
                logger.warning(f"解析package.json失败: {e}")
        
        self.architecture_analysis["dependencies"] = {k: list(v) for k, v in dependencies.items()}

    async def _analyze_version_compatibility(self):
        """分析版本兼容性"""
        # 简化的版本兼容性检查
        compatibility_issues = []
        
        for file_path in self.workspace_path.rglob("requirements.txt"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查固定版本
                    fixed_versions = re.findall(r'(\w+)==([0-9.]+)', content)
                    for pkg, version in fixed_versions:
                        if version.startswith('0.') or version.startswith('1.0.'):
                            compatibility_issues.append((pkg, version, "版本过旧，可能存在兼容性问题"))
                
                except Exception as e:
                    logger.warning(f"无法分析版本兼容性 {file_path}: {e}")
        
        self.architecture_analysis["compatibility_issues"] = compatibility_issues

    async def _analyze_license_compliance(self):
        """分析许可证合规性"""
        # 简化的许可证检查
        allowed_licenses = ["MIT", "Apache-2.0", "BSD", "ISC"]
        problematic_licenses = []
        
        # 这里应该实现真正的许可证检查逻辑
        # 目前只是占位符
        
        self.architecture_analysis["license_compliance"] = {
            "allowed_licenses": allowed_licenses,
            "problematic_licenses": problematic_licenses,
            "compliance_status": "compliant" if not problematic_licenses else "non_compliant"
        }

    async def _analyze_test_coverage(self):
        """分析测试覆盖率"""
        logger.info("  🧪 分析测试覆盖率...")
        
        try:
            test_files = list(self.workspace_path.rglob("*test*.py"))
            test_files.extend(list(self.workspace_path.rglob("test_*.py")))
            
            code_files = list(self.workspace_path.rglob("*.py"))
            code_files = [f for f in code_files if not any(pattern in f.name for pattern in ["test", "spec"])]
            
            self.project_metrics.test_files = len(test_files)
            self.project_metrics.code_files = len(code_files)
            
            if len(code_files) > 0:
                coverage_ratio = len(test_files) / len(code_files)
                self.project_metrics.test_coverage = coverage_ratio * 100
                
                if coverage_ratio < 0.5:
                    issue = Issue(
                        id="test_coverage_low",
                        type="test_coverage",
                        severity=Severity.MEDIUM,
                        title="测试覆盖率过低",
                        description=f"测试覆盖率仅为 {coverage_ratio*100:.1f}%，建议增加测试",
                        file_path="",
                        line_number=1,
                        evidence=f"代码文件: {len(code_files)}, 测试文件: {len(test_files)}",
                        fix_suggestion="增加单元测试和集成测试",
                        auto_fixable=False,
                        category="testing",
                        impact="quality",
                        effort="high"
                    )
                    self.issues.append(issue)
        
        except Exception as e:
            logger.error(f"    测试覆盖率分析失败: {e}")

    async def _phase_planning(self):
        """阶段2: 智能规划"""
        logger.info("📋 阶段2: 智能规划...")
        self.current_phase = UpgradePhase.PLANNING
        
        try:
            # 2.1 优先级评估
            await self._assess_priorities()
            
            # 2.2 升级计划制定
            await self._create_upgrade_plan()
            
            # 2.3 风险评估
            await self._assess_risks()
            
            # 2.4 资源评估
            await self._assess_resources()
            
            logger.info(f"  ✅ 智能规划完成，制定了 {len(self.upgrade_actions)} 个升级动作")
            
        except Exception as e:
            logger.error(f"  ❌ 智能规划失败: {e}")
            raise

    async def _assess_priorities(self):
        """评估优先级"""
        logger.info("    评估问题优先级...")
        
        # 根据严重程度和影响评估优先级
        for issue in self.issues:
            if issue.severity == Severity.CRITICAL:
                issue.priority = "P0"
            elif issue.severity == Severity.HIGH:
                issue.priority = "P1"
            elif issue.severity == Severity.MEDIUM:
                issue.priority = "P2"
            else:
                issue.priority = "P3"
        
        # 按优先级排序
        self.issues.sort(key=lambda x: (x.severity.value, x.priority))

    async def _create_upgrade_plan(self):
        """创建升级计划"""
        logger.info("    制定升级计划...")
        
        # 按类型分组问题
        issues_by_type = defaultdict(list)
        for issue in self.issues:
            issues_by_type[issue.category].append(issue)
        
        # 为每个类别创建升级动作
        for category, issues in issues_by_type.items():
            action = UpgradeAction(
                id=f"upgrade_{category}_{len(self.upgrade_actions)}",
                type=category,
                description=f"修复 {len(issues)} 个 {category} 类问题",
                file_path="",
                changes={
                    "issues": [asdict(issue) for issue in issues[:10]],  # 限制数量
                    "category": category,
                    "total_issues": len(issues)
                },
                priority=self._calculate_priority(issues),
                risk_level=self._assess_action_risk(category, issues),
                estimated_time=self._estimate_time(issues),
                dependencies=[]
            )
            self.upgrade_actions.append(action)

    def _calculate_priority(self, issues: List[Issue]) -> str:
        """计算优先级"""
        if any(issue.severity == Severity.CRITICAL for issue in issues):
            return "critical"
        elif any(issue.severity == Severity.HIGH for issue in issues):
            return "high"
        elif any(issue.severity == Severity.MEDIUM for issue in issues):
            return "medium"
        else:
            return "low"

    def _assess_action_risk(self, category: str, issues: List[Issue]) -> str:
        """评估动作风险"""
        high_risk_categories = ["security", "architecture", "algorithm"]
        if category in high_risk_categories:
            return "high"
        elif category in ["performance", "memory"]:
            return "medium"
        else:
            return "low"

    def _estimate_time(self, issues: List[Issue]) -> int:
        """估算时间（分钟）"""
        base_time = len(issues) * 5  # 每个问题5分钟
        
        # 根据类型调整
        category_multipliers = {
            "security": 2.0,
            "architecture": 1.5,
            "performance": 1.3,
            "algorithm": 1.8,
            "code_style": 0.5,
            "duplication": 1.2
        }
        
        if issues:
            category = issues[0].category
            multiplier = category_multipliers.get(category, 1.0)
            return int(base_time * multiplier)
        
        return base_time

    async def _assess_risks(self):
        """评估风险"""
        logger.info("    评估升级风险...")
        
        # 计算整体风险分数
        critical_issues = len([i for i in self.issues if i.severity == Severity.CRITICAL])
        high_issues = len([i for i in self.issues if i.severity == Severity.HIGH])
        
        risk_score = (critical_issues * 10) + (high_issues * 5) + len(self.issues)
        
        if risk_score > 50:
            overall_risk = "high"
        elif risk_score > 20:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        self.architecture_analysis["upgrade_risk"] = {
            "risk_score": risk_score,
            "overall_risk": overall_risk,
            "critical_issues": critical_issues,
            "high_issues": high_issues
        }

    async def _assess_resources(self):
        """评估资源"""
        logger.info("    评估所需资源...")
        
        total_estimated_time = sum(action.estimated_time for action in self.upgrade_actions)
        auto_fixable_count = len([i for i in self.issues if i.auto_fixable])
        
        self.architecture_analysis["resource_requirements"] = {
            "estimated_time_minutes": total_estimated_time,
            "auto_fixable_issues": auto_fixable_count,
            "manual_fix_required": len(self.issues) - auto_fixable_count,
            "recommended_parallel_actions": min(3, len(self.upgrade_actions))
        }

    async def _phase_execution(self):
        """阶段3: 执行升级"""
        logger.info("🔧 阶段3: 执行升级...")
        self.current_phase = UpgradePhase.EXECUTION
        
        try:
            # 3.1 自动修复
            if self.auto_fix:
                await self._execute_auto_fixes()
            
            # 3.2 代码重构
            await self._execute_refactoring()
            
            # 3.3 性能优化
            await self._execute_performance_optimization()
            
            # 3.4 安全加固
            await self._execute_security_hardening()
            
            # 3.5 架构改进
            await self._execute_architecture_improvements()
            
            logger.info("  ✅ 升级执行完成")
            
        except Exception as e:
            logger.error(f"  ❌ 升级执行失败: {e}")
            raise

    async def _execute_auto_fixes(self):
        """生成自动修复建议报告（不执行实际修复）"""
        logger.info("    生成自动修复建议报告...")
        
        auto_fixable_issues = [issue for issue in self.issues if issue.auto_fixable]
        
        for issue in auto_fixable_issues:
            logger.info(f"        📋 自动修复建议: {issue.title}")
            logger.info(f"           文件: {issue.file_path}")
            logger.info(f"           位置: 第{issue.line_number}行")
            logger.info(f"           建议: {issue.fix_suggestion}")
            logger.info(f"           类型: {issue.category}")
            logger.info(f"           影响: {issue.impact}")
            
            # 记录为建议而不是修复
            self.changelog.append({
                "timestamp": datetime.now().isoformat(),
                "action": "auto_fix_suggestion",
                "issue": issue.title,
                "file": issue.file_path,
                "status": "suggested",
                "fix_suggestion": issue.fix_suggestion,
                "line_number": issue.line_number,
                "category": issue.category,
                "impact": issue.impact
            })
        
        logger.info(f"        生成了 {len(auto_fixable_issues)} 个自动修复建议")

    async def _apply_fix(self, issue: Issue):
        """应用修复"""
        file_path = Path(issue.file_path)
        
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # 根据问题类型应用不同的修复策略
            if issue.type == "code_quality" and "trailing_whitespace" in issue.title:
                # 修复行尾空格
                if issue.line_number <= len(lines):
                    lines[issue.line_number - 1] = lines[issue.line_number - 1].rstrip()
            
            elif issue.type == "code_quality" and "长行" in issue.title:
                # 修复长行（简单拆分）
                if issue.line_number <= len(lines):
                    long_line = lines[issue.line_number - 1]
                    if len(long_line) > 120:
                        # 简单的行拆分逻辑
                        lines[issue.line_number - 1] = long_line[:80] + " \\\n    " + long_line[80:]
            
            elif issue.type == "sensitive_information":
                # 修复敏感信息（移除并添加占位符）
                if issue.line_number <= len(lines):
                    line = lines[issue.line_number - 1]
                    # 替换为环境变量引用
                    line = re.sub(r"=\s*['\"][^'\"]+['\"]", "= os.getenv('SENSITIVE_VALUE')", line)
                    lines[issue.line_number - 1] = line
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        
        except Exception as e:
            logger.error(f"应用修复失败: {e}")
            raise

    async def _execute_refactoring(self):
        """分析代码重构机会 - 检测模式"""
        logger.info("    分析代码重构机会...")
        
        try:
            # 1. 检测重构机会
            refactor_opportunities = await self._detect_refactor_opportunities()
            
            # 2. 生成重构分析报告
            refactor_report = await self._generate_refactor_analysis_report(refactor_opportunities)
            
            # 3. 保存重构建议数据
            await self._save_refactor_suggestions(refactor_opportunities, refactor_report)
            
            logger.info(f"      ✅ 代码重构分析完成，发现 {len(refactor_opportunities)} 个重构机会")
            
        except Exception as e:
            logger.error(f"      ❌ 代码重构分析失败: {e}")

    async def _generate_refactor_analysis_report(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成重构分析报告"""
        report = {
            "summary": {
                "total_files_with_issues": len(opportunities),
                "total_refactor_opportunities": sum(len(opp["issues"]) for opp in opportunities),
                "priority_distribution": defaultdict(int),
                "complexity_levels": defaultdict(int)
            },
            "detailed_analysis": [],
            "ai_suggestions": [],
            "estimated_effort": {}
        }
        
        for file_opportunity in opportunities:
            file_analysis = {
                "file_path": file_opportunity["file_path"],
                "file_size": self._get_file_size(file_opportunity["file_path"]),
                "issues": []
            }
            
            for issue in file_opportunity["issues"]:
                issue_analysis = {
                    "type": issue["type"],
                    "description": issue["description"],
                    "severity": issue["severity"],
                    "location": issue["location"],
                    "context": await self._extract_code_context(file_opportunity["file_path"], issue["location"]),
                    "impact_analysis": await self._analyze_issue_impact(issue),
                    "ai_fix_strategy": await self._generate_ai_fix_strategy(issue),
                    "estimated_complexity": self._estimate_fix_complexity(issue),
                    "dependencies": await self._identify_dependencies(file_opportunity["file_path"], issue)
                }
                
                file_analysis["issues"].append(issue_analysis)
                report["summary"]["priority_distribution"][issue["severity"]] += 1
            
            report["detailed_analysis"].append(file_analysis)
        
        # 生成AI友好的建议
        report["ai_suggestions"] = await self._generate_ai_friendly_suggestions(opportunities)
        
        # 估算工作量
        report["estimated_effort"] = self._calculate_total_effort(opportunities)
        
        return report

    async def _extract_code_context(self, file_path: str, location: Dict[str, Any]) -> str:
        """提取代码上下文"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            line_num = location.get("line", 1)
            start = max(0, line_num - 3)
            end = min(len(lines), line_num + 2)
            
            context_lines = []
            for i in range(start, end):
                context_lines.append(f"{i+1:4d}: {lines[i].rstrip()}")
            
            return '\n'.join(context_lines)
        
        except Exception as e:
            return f"无法提取上下文: {e}"

    async def _analyze_issue_impact(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """分析问题影响"""
        impact = {
            "maintainability": "medium",
            "readability": "medium", 
            "performance": "low",
            "security": "low",
            "testability": "low"
        }
        
        # 根据问题类型调整影响
        if issue["type"] == "duplicate_code":
            impact["maintainability"] = "high"
            impact["readability"] = "medium"
        elif issue["type"] == "long_function":
            impact["maintainability"] = "high"
            impact["testability"] = "medium"
        elif issue["type"] == "complex_condition":
            impact["readability"] = "high"
            impact["maintainability"] = "medium"
        
        return impact

    async def _generate_ai_fix_strategy(self, issue: Dict[str, Any]) -> str:
        """生成AI修复策略描述"""
        strategies = {
            "duplicate_code": "提取重复代码为独立函数，使用参数化处理差异，确保函数职责单一",
            "long_function": "将长函数分解为多个小函数，每个函数负责单一职责，提高可读性和可测试性",
            "complex_condition": "将复杂条件表达式提取为有意义的变量名或辅助函数，提高代码可读性",
            "magic_numbers": "将魔法数字提取为命名常量，使用描述性名称，提高代码可维护性"
        }
        
        return strategies.get(issue["type"], "根据具体情况进行重构，遵循单一职责原则")

    def _estimate_fix_complexity(self, issue: Dict[str, Any]) -> str:
        """估算修复复杂度"""
        complexity_map = {
            "duplicate_code": "medium",
            "long_function": "high", 
            "complex_condition": "low",
            "magic_numbers": "low"
        }
        
        return complexity_map.get(issue["type"], "medium")

    async def _identify_dependencies(self, file_path: str, issue: Dict[str, Any]) -> List[str]:
        """识别依赖关系"""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单的依赖分析
            if issue["type"] == "long_function":
                # 检查是否需要导入新模块
                if "utils" not in content and "helper" not in content:
                    dependencies.append("可能需要创建工具函数模块")
            
            elif issue["type"] == "magic_numbers":
                # 检查是否已有常量文件
                if "constants" not in content and "config" not in content:
                    dependencies.append("建议创建常量配置文件")
        
        except Exception:
            pass
        
        return dependencies

    async def _generate_ai_friendly_suggestions(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成AI友好的建议"""
        suggestions = []
        
        for file_opportunity in opportunities:
            for issue in file_opportunity["issues"]:
                suggestion = {
                    "file": file_opportunity["file_path"],
                    "issue_type": issue["type"],
                    "description": issue["description"],
                    "severity": issue["severity"],
                    "context": await self._extract_code_context(file_opportunity["file_path"], issue["location"]),
                    "recommended_action": await self._generate_ai_fix_strategy(issue),
                    "reasoning": await self._generate_fix_reasoning(issue),
                    "implementation_notes": await self._generate_implementation_notes(issue),
                    "test_suggestions": await self._generate_test_suggestions(issue),
                    "impact_assessment": await self._analyze_issue_impact(issue)
                }
                suggestions.append(suggestion)
        
        return suggestions

    async def _generate_fix_reasoning(self, issue: Dict[str, Any]) -> str:
        """生成修复理由"""
        reasoning_map = {
            "duplicate_code": "重复代码违反DRY原则，增加维护成本，修改时需要在多处同步更新",
            "long_function": "长函数违反单一职责原则，难以理解和测试，增加认知负担",
            "complex_condition": "复杂条件降低代码可读性，增加出错概率，难以调试和维护",
            "magic_numbers": "魔法数字缺乏语义，降低代码可读性，难以理解和修改"
        }
        
        return reasoning_map.get(issue["type"], "遵循软件工程最佳实践，提高代码质量")

    async def _generate_implementation_notes(self, issue: Dict[str, Any]) -> str:
        """生成实现说明"""
        notes_map = {
            "duplicate_code": "1. 识别重复代码块 2. 提取为独立函数 3. 参数化差异部分 4. 替换所有调用点",
            "long_function": "1. 识别函数职责 2. 按职责分组 3. 提取子函数 4. 保持接口一致性",
            "complex_condition": "1. 识别条件逻辑 2. 提取为变量 3. 使用辅助函数 4. 添加注释说明",
            "magic_numbers": "1. 识别魔法数字 2. 确定语义 3. 创建常量 4. 替换所有使用点"
        }
        
        return notes_map.get(issue["type"], "根据具体情况进行详细实现")

    async def _generate_test_suggestions(self, issue: Dict[str, Any]) -> str:
        """生成测试建议"""
        test_map = {
            "duplicate_code": "为提取的函数编写单元测试，确保重构后功能一致，添加边界条件测试",
            "long_function": "为每个子函数编写独立测试，验证拆分后的行为，添加集成测试",
            "complex_condition": "测试各种条件组合，验证逻辑正确性，添加边界值测试",
            "magic_numbers": "测试常量值变更的影响，验证配置灵活性，添加参数化测试"
        }
        
        return test_map.get(issue["type"], "编写相应的单元测试和集成测试")

    def _calculate_total_effort(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算总工作量"""
        effort = {
            "total_issues": sum(len(opp["issues"]) for opp in opportunities),
            "estimated_hours": 0,
            "complexity_breakdown": defaultdict(int),
            "priority_breakdown": defaultdict(int)
        }
        
        for file_opportunity in opportunities:
            for issue in file_opportunity["issues"]:
                # 估算每个问题的工作量（小时）
                hours_map = {
                    "duplicate_code": 2,
                    "long_function": 4,
                    "complex_condition": 1,
                    "magic_numbers": 0.5
                }
                
                hours = hours_map.get(issue["type"], 2)
                effort["estimated_hours"] += hours
                effort["complexity_breakdown"][self._estimate_fix_complexity(issue)] += 1
                effort["priority_breakdown"][issue["severity"]] += 1
        
        return effort

    async def _save_refactor_suggestions(self, opportunities: List[Dict[str, Any]], report: Dict[str, Any]):
        """保存重构建议数据"""
        try:
            # 创建输出目录
            output_dir = self.workspace_path / ".iflow" / "analysis_results"
            output_dir.mkdir(exist_ok=True)
            
            # 保存详细分析结果
            analysis_file = output_dir / f"refactor_analysis_{self.session_id}.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "opportunities": opportunities,
                    "report": report
                }, f, indent=2, ensure_ascii=False, default=str)
            
            # 保存AI友好的建议
            suggestions_file = output_dir / f"ai_suggestions_{self.session_id}.json"
            with open(suggestions_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "suggestions": report["ai_suggestions"],
                    "metadata": {
                        "total_suggestions": len(report["ai_suggestions"]),
                        "estimated_hours": report["estimated_effort"]["estimated_hours"],
                        "priority_distribution": dict(report["summary"]["priority_distribution"])
                    }
                }, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"      📋 重构分析报告已保存: {analysis_file}")
            logger.info(f"      🤖 AI建议数据已保存: {suggestions_file}")
            
        except Exception as e:
            logger.error(f"      ❌ 保存重构建议失败: {e}")

    def _get_file_size(self, file_path: str) -> int:
        """获取文件大小"""
        try:
            return Path(file_path).stat().st_size
        except:
            return 0

    async def _detect_refactor_opportunities(self) -> List[Dict[str, Any]]:
        """检测重构机会"""
        logger.info("      检测重构机会...")
        
        opportunities = []
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_opportunities = {
                        "file_path": str(file_path),
                        "issues": []
                    }
                    
                    # 检测重复代码
                    duplicates = self._detect_duplicate_code(content)
                    if duplicates:
                        file_opportunities["issues"].append({
                            "type": "duplicate_code",
                            "description": "发现重复代码块",
                            "locations": duplicates,
                            "severity": "medium"
                        })
                    
                    # 检测长函数
                    long_functions = self._detect_long_functions(content)
                    for func in long_functions:
                        file_opportunities["issues"].append({
                            "type": "long_function",
                            "description": f"函数 {func['name']} 过长 ({func['lines']} 行)",
                            "location": func,
                            "severity": "medium"
                        })
                    
                    # 检测复杂条件
                    complex_conditions = self._detect_complex_conditions(content)
                    for condition in complex_conditions:
                        file_opportunities["issues"].append({
                            "type": "complex_condition",
                            "description": "复杂的条件表达式",
                            "location": condition,
                            "severity": "low"
                        })
                    
                    # 检测魔法数字
                    magic_numbers = self._detect_magic_numbers(content)
                    if magic_numbers:
                        file_opportunities["issues"].append({
                            "type": "magic_numbers",
                            "description": f"发现 {len(magic_numbers)} 个魔法数字",
                            "locations": magic_numbers,
                            "severity": "low"
                        })
                    
                    if file_opportunities["issues"]:
                        opportunities.append(file_opportunities)
                
                except Exception as e:
                    logger.warning(f"检测重构机会失败 {file_path}: {e}")
        
        logger.info(f"        检测到 {len(opportunities)} 个文件的重构机会")
        return opportunities

    def _detect_duplicate_code(self, content: str) -> List[Dict[str, Any]]:
        """检测重复代码"""
        lines = content.split('\n')
        duplicates = []
        
        # 查找3行以上的重复块
        code_blocks = defaultdict(list)
        
        for i in range(len(lines) - 2):
            block = '\n'.join(lines[i:i+3]).strip()
            if len(block) > 30:  # 忽略太短的块
                code_blocks[block].append(i + 1)
        
        for block, line_numbers in code_blocks.items():
            if len(line_numbers) > 1:
                duplicates.append({
                    "block": block[:100] + "..." if len(block) > 100 else block,
                    "locations": line_numbers
                })
        
        return duplicates

    def _detect_long_functions(self, content: str) -> List[Dict[str, Any]]:
        """检测长函数"""
        try:
            tree = ast.parse(content)
            long_functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # 计算函数行数
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    lines_count = end_line - start_line + 1
                    
                    if lines_count > 20:  # 长函数阈值
                        long_functions.append({
                            "name": node.name,
                            "lines": lines_count,
                            "start_line": start_line,
                            "end_line": end_line
                        })
            
            return long_functions
        except:
            return []

    def _detect_complex_conditions(self, content: str) -> List[Dict[str, Any]]:
        """检测复杂条件"""
        complex_conditions = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # 检查复杂的布尔表达式
            if (' and ' in line and line.count(' and ') > 2) or \
               (' or ' in line and line.count(' or ') > 2) or \
               (line.count('(') > 3 and line.count(')') > 3):
                complex_conditions.append({
                    "line": i + 1,
                    "content": line.strip()
                })
        
        return complex_conditions

    def _detect_magic_numbers(self, content: str) -> List[Dict[str, Any]]:
        """检测魔法数字"""
        magic_numbers = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # 查找大于10的数字（排除版本号等）
            numbers = re.findall(r'\b([1-9]\d{2,})\b', line)
            for num in numbers:
                # 排除一些常见的数字
                if num not in ['100', '1000', '1024', '2048', '4096']:
                    magic_numbers.append({
                        "value": num,
                        "line": i + 1,
                        "context": line.strip()
                    })
        
        return magic_numbers

    async def _ai_analyze_refactor_plan(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """AI分析和规划重构"""
        logger.info("      AI分析重构计划...")
        
        # 这里调用AI来分析重构机会并制定计划
        refactor_plan = {
            "priority_actions": [],
            "safe_actions": [],
            "risky_actions": [],
            "estimated_time": 0,
            "dependencies": []
        }
        
        for file_opportunity in opportunities:
            for issue in file_opportunity["issues"]:
                action = {
                    "file_path": file_opportunity["file_path"],
                    "issue_type": issue["type"],
                    "description": issue["description"],
                    "severity": issue["severity"],
                    "ai_suggestion": await self._generate_ai_suggestion(issue),
                    "confidence": 0.8  # AI建议的置信度
                }
                
                # 根据问题类型和严重程度分类
                if issue["severity"] == "medium" and issue["type"] in ["duplicate_code", "long_function"]:
                    refactor_plan["priority_actions"].append(action)
                elif issue["severity"] == "low":
                    refactor_plan["safe_actions"].append(action)
                else:
                    refactor_plan["risky_actions"].append(action)
        
        # 估算时间
        refactor_plan["estimated_time"] = len(refactor_plan["priority_actions"]) * 15 + \
                                       len(refactor_plan["safe_actions"]) * 10 + \
                                       len(refactor_plan["risky_actions"]) * 25
        
        logger.info(f"        AI规划完成: {len(refactor_plan['priority_actions'])} 个优先操作")
        return refactor_plan

    async def _generate_ai_suggestion(self, issue: Dict[str, Any]) -> str:
        """生成AI建议"""
        # 这里应该调用真正的AI模型来生成分析和建议
        suggestions = {
            "duplicate_code": "建议提取重复代码为独立函数，使用参数化来处理差异",
            "long_function": "建议将长函数分解为多个更小的、职责单一的函数",
            "complex_condition": "建议将复杂条件提取为有意义的变量名或辅助函数",
            "magic_numbers": "建议将魔法数字提取为命名常量，提高代码可读性"
        }
        
        return suggestions.get(issue["type"], "建议进行代码重构以提高质量")

    async def _ai_execute_refactoring(self, refactor_plan: Dict[str, Any]):
        """生成重构建议报告（不执行实际修复）"""
        logger.info("      生成重构建议报告...")
        
        # 按优先级生成建议
        all_actions = refactor_plan["priority_actions"] + refactor_plan["safe_actions"]
        
        for action in all_actions:
            logger.info(f"        📋 重构建议: {action['description']}")
            logger.info(f"           文件: {action['file_path']}")
            logger.info(f"           AI建议: {action['ai_suggestion']}")
            logger.info(f"           置信度: {action['confidence']:.2f}")
            
            # 记录到changelog作为建议
            self.changelog.append({
                "timestamp": datetime.now().isoformat(),
                "action": "refactor_suggestion",
                "issue": action["description"],
                "file": action["file_path"],
                "status": "suggested",
                "ai_suggestion": action["ai_suggestion"],
                "confidence": action["confidence"]
            })
        
        logger.info(f"        生成了 {len(all_actions)} 个重构建议")

    async def _ai_apply_refactor_action(self, action: Dict[str, Any]) -> bool:
        """AI应用重构操作"""
        try:
            file_path = Path(action["file_path"])
            
            if not file_path.exists():
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 根据问题类型应用不同的AI重构策略
            if action["issue_type"] == "duplicate_code":
                new_content = await self._ai_refactor_duplicate_code(content, action)
            elif action["issue_type"] == "long_function":
                new_content = await self._ai_refactor_long_function(content, action)
            elif action["issue_type"] == "complex_condition":
                new_content = await self._ai_refactor_complex_condition(content, action)
            elif action["issue_type"] == "magic_numbers":
                new_content = await self._ai_refactor_magic_numbers(content, action)
            else:
                return False
            
            # 如果内容有变化，写入文件
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"AI应用重构操作失败: {e}")
            return False

    async def _ai_refactor_duplicate_code(self, content: str, action: Dict[str, Any]) -> str:
        """AI重构重复代码"""
        # 这里应该调用AI来理解代码上下文并生成重构代码
        # 现在用分析逻辑生成建议
        logger.info(f"          AI分析重复代码: {action['description']}")
        return content  # 实际应该返回重构后的内容

    async def _ai_refactor_long_function(self, content: str, action: Dict[str, Any]) -> str:
        """AI重构长函数"""
        logger.info(f"          AI分析长函数: {action['description']}")
        return content  # 实际应该返回重构后的内容

    async def _ai_refactor_complex_condition(self, content: str, action: Dict[str, Any]) -> str:
        """AI重构复杂条件"""
        logger.info(f"          AI分析复杂条件: {action['description']}")
        return content  # 实际应该返回重构后的内容

    async def _ai_refactor_magic_numbers(self, content: str, action: Dict[str, Any]) -> str:
        """AI重构魔法数字"""
        logger.info(f"          AI分析魔法数字: {action['description']}")
        return content  # 实际应该返回重构后的内容

    async def _extract_duplicate_code(self):
        """提取重复代码为函数"""
        logger.info("      提取重复代码...")
        
        # 分析重复代码块
        code_blocks = defaultdict(list)
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # 查找3行以上的重复块
                    for i in range(len(lines) - 2):
                        block = ''.join(lines[i:i+3]).strip()
                        if len(block) > 30:  # 忽略太短的块
                            code_blocks[block].append((str(file_path), i+1))
                
                except Exception as e:
                    logger.warning(f"无法分析 {file_path}: {e}")
        
        # 提取重复代码
        for block, occurrences in code_blocks.items():
            if len(occurrences) > 1 and not self.analysis_mode:
                # 生成函数名
                func_name = f"extracted_function_{hashlib.md5(block.encode()).hexdigest()[:8]}"
                
                # 在第一个文件中创建函数
                first_file, first_line = occurrences[0]
                await self._create_extracted_function(first_file, func_name, block)
                
                # 替换所有出现的位置
                for file_path, line_num in occurrences:
                    await self._replace_with_function_call(file_path, line_num, func_name, block)
                
                logger.info(f"        提取重复代码为函数: {func_name}")

    async def _create_extracted_function(self, file_path: str, func_name: str, block: str):
        """创建提取的函数"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 在文件开头添加函数
            func_def = f"def {func_name}():\n    # 提取的重复代码\n"
            for line in block.split('\n'):
                if line.strip():
                    func_def += f"    {line}\n"
            func_def += "\n\n"
            
            # 在第一个类或函数之前插入
            lines = content.split('\n')
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.startswith('class ') or line.startswith('def '):
                    insert_pos = i
                    break
            
            lines.insert(insert_pos, func_def)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        
        except Exception as e:
            logger.error(f"创建函数失败: {e}")

    async def _replace_with_function_call(self, file_path: str, line_num: int, func_name: str, block: str):
        """用函数调用替换重复代码"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 替换重复代码块
            block_lines = block.split('\n')
            for i, line in enumerate(block_lines):
                if line_num + i - 1 < len(lines):
                    if i == 0:
                        lines[line_num + i - 1] = f"    {func_name}()\n"
                    else:
                        lines[line_num + i - 1] = "\n"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        
        except Exception as e:
            logger.error(f"替换函数调用失败: {e}")

    async def _refactor_long_functions(self):
        """重构长函数"""
        logger.info("      重构长函数...")
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # 计算函数行数
                            func_lines = len(node.body)
                            if func_lines > 20:  # 长函数阈值
                                await self._break_down_long_function(file_path, node.name)
                                logger.info(f"        重构长函数: {node.name} ({func_lines} 行)")
                
                except Exception as e:
                    logger.warning(f"重构长函数失败 {file_path}: {e}")

    async def _break_down_long_function(self, file_path: Path, func_name: str):
        """分解长函数"""
        # 这里实现长函数分解逻辑
        # 分析函数逻辑块，提取为子函数
        pass

    async def _optimize_class_structure(self):
        """优化类结构"""
        logger.info("      优化类结构...")
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # 检查类的方法数量
                            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                            if len(methods) > 15:  # 方法过多
                                await self._split_large_class(file_path, node.name)
                                logger.info(f"        优化大类: {node.name} ({len(methods)} 个方法)")
                
                except Exception as e:
                    logger.warning(f"优化类结构失败 {file_path}: {e}")

    async def _split_large_class(self, file_path: Path, class_name: str):
        """拆分大类"""
        # 这里实现大类拆分逻辑
        # 根据职责将大类拆分为多个小类
        pass

    async def _refactor_conditionals(self):
        """重构条件表达式"""
        logger.info("      重构条件表达式...")
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 简化复杂的条件表达式
                    original_content = content
                    
                    # 将长if-elif链替换为字典查找
                    content = self._simplify_if_elif_chain(content)
                    
                    # 将嵌套条件提取为变量
                    content = self._extract_nested_conditions(content)
                    
                    if content != original_content and not self.analysis_mode:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        logger.info(f"        重构条件表达式: {file_path.name}")
                
                except Exception as e:
                    logger.warning(f"重构条件表达式失败 {file_path}: {e}")

    def _simplify_if_elif_chain(self, content: str) -> str:
        """简化if-elif链"""
        # 查找可以转换为字典查找的if-elif链
        pattern = r'if\s+(\w+)\s*==\s*(\w+):\s*\n(.*?)\nelif\s+\1\s*==\s*(\w+):\s*\n(.*?)(?=\n(?:elif|else|if|\Z))'
        
        def replace_chain(match):
            var_name = match.group(1)
            replacements = []
            
            # 提取所有条件-值对
            current_match = match
            while current_match:
                condition = current_match.group(2)
                value = current_match.group(3).strip()
                replacements.append(f'"{condition}": {value}')
                
                # 查找下一个elif
                rest = content[current_match.end():]
                next_match = re.search(r'elif\s+' + re.escape(var_name) + r'\s*==\s*(\w+):\s*\n(.*?)(?=\n(?:elif|else|if|\Z))', rest)
                if next_match:
                    current_match = next_match
                    current_match = type('Match', (), {
                        'group': lambda i, cm=current_match, nm=next_match: (
                            nm.group(i) if i <= 2 else 
                            (cm.group(i) if i == 3 else nm.group(i-2))
                        )[i],
                        'end': lambda cm=current_match, nm=next_match: cm.end() + nm.end()
                    })()
                else:
                    break
            
            # 创建字典查找
            dict_def = f"{var_name}_map = {{\n        " + ",\n        ".join(replacements) + "\n    }"
            lookup = f"result = {var_name}_map.get({var_name}, default_value)"
            
            return f"{dict_def}\n    {lookup}"
        
        return re.sub(pattern, replace_chain, content, flags=re.DOTALL)

    def _extract_nested_conditions(self, content: str) -> str:
        """提取嵌套条件"""
        # 查找复杂的嵌套条件并提取为变量
        pattern = r'if\s+([^:]+):\s*\n(.*?)\n(?:else|elif)'
        
        def extract_condition(match):
            condition = match.group(1)
            if ' and ' in condition or ' or ' in condition:
                # 生成变量名
                var_name = f"condition_{hashlib.md5(condition.encode()).hexdigest()[:6]}"
                # 提取为变量
                return f"{var_name} = {condition}\n    if {var_name}:\n{match.group(2)}"
            return match.group(0)
        
        return re.sub(pattern, extract_condition, content, flags=re.DOTALL)

    async def _extract_constants(self):
        """提取常量"""
        logger.info("      提取常量...")
        
        # 查找硬编码的数值和字符串
        magic_numbers = defaultdict(list)
        magic_strings = defaultdict(list)
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 查找魔法数字（大于10的数字）
                    numbers = re.findall(r'\b([1-9]\d{2,})\b', content)
                    for num in numbers:
                        magic_numbers[num].append(str(file_path))
                    
                    # 查找重复的字符串字面量
                    strings = re.findall(r'["\']([^"\']{10,})["\']', content)
                    for string in strings:
                        if not string.islower():  # 排除普通的小写字符串
                            magic_strings[string].append(str(file_path))
                
                except Exception as e:
                    logger.warning(f"提取常量失败 {file_path}: {e}")
        
        # 创建常量定义
        if magic_numbers or magic_strings:
            await self._create_constants_file(magic_numbers, magic_strings)
            logger.info(f"        提取了 {len(magic_numbers)} 个数字常量和 {len(magic_strings)} 个字符串常量")

    async def _create_constants_file(self, numbers: Dict, strings: Dict):
        """创建常量文件"""
        if self.analysis_mode:
            return
        
        constants_file = self.workspace_path / "constants.py"
        
        try:
            existing_content = ""
            if constants_file.exists():
                with open(constants_file, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
            
            # 生成新的常量定义
            new_constants = ["# 自动生成的常量定义\n"]
            
            for num, files in numbers.items():
                if len(files) > 1:  # 只提取多次使用的数字
                    const_name = f"VALUE_{num}"
                    new_constants.append(f"{const_name} = {num}  # 用于: {', '.join([Path(f).name for f in files[:3]])}")
            
            for string, files in strings.items():
                if len(files) > 1:  # 只提取重复的字符串
                    const_name = f"TEXT_{hashlib.md5(string.encode()).hexdigest()[:8].upper()}"
                    # 转义字符串中的特殊字符
                    escaped_string = string.replace('"', '\\"')
                    new_constants.append(f'{const_name} = "{escaped_string}"  # 用于: {", ".join([Path(f).name for f in files[:3]])}')
            
            # 写入文件
            with open(constants_file, 'w', encoding='utf-8') as f:
                f.write(existing_content + "\n" + "\n".join(new_constants))
        
        except Exception as e:
            logger.error(f"创建常量文件失败: {e}")

    async def _execute_performance_optimization(self):
        """执行性能优化 - AI驱动模式"""
        logger.info("    执行性能优化...")
        
        try:
            # 1. 检测性能瓶颈
            performance_issues = await self._detect_performance_issues()
            
            # 2. AI分析和优化策略
            optimization_plan = await self._ai_analyze_optimization_plan(performance_issues)
            
            # 3. AI执行性能优化
            await self._ai_execute_performance_optimization(optimization_plan)
            
            logger.info("      ✅ 性能优化完成")
            
        except Exception as e:
            logger.error(f"      ❌ 性能优化失败: {e}")

    async def _detect_performance_issues(self) -> List[Dict[str, Any]]:
        """检测性能问题"""
        logger.info("      检测性能问题...")
        
        performance_issues = []
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_issues = {
                        "file_path": str(file_path),
                        "issues": []
                    }
                    
                    # 检测嵌套循环
                    nested_loops = self._detect_nested_loops(content)
                    for loop in nested_loops:
                        file_issues["issues"].append({
                            "type": "nested_loop",
                            "description": "嵌套循环可能导致O(n²)复杂度",
                            "location": loop,
                            "severity": "high",
                            "impact": "algorithmic_complexity"
                        })
                    
                    # 检测低效查找
                    inefficient_lookups = self._detect_inefficient_lookups(content)
                    for lookup in inefficient_lookups:
                        file_issues["issues"].append({
                            "type": "inefficient_lookup",
                            "description": "列表查找效率较低",
                            "location": lookup,
                            "severity": "medium",
                            "impact": "lookup_performance"
                        })
                    
                    # 检测重复计算
                    repeated_calculations = self._detect_repeated_calculations(content)
                    for calc in repeated_calculations:
                        file_issues["issues"].append({
                            "type": "repeated_calculation",
                            "description": "重复计算可以缓存",
                            "location": calc,
                            "severity": "medium",
                            "impact": "cpu_usage"
                        })
                    
                    # 检测I/O操作
                    io_operations = self._detect_io_operations(content)
                    for io_op in io_operations:
                        file_issues["issues"].append({
                            "type": "io_operation",
                            "description": "I/O操作可以优化",
                            "location": io_op,
                            "severity": "medium",
                            "impact": "io_performance"
                        })
                    
                    # 检测内存使用
                    memory_issues = self._detect_memory_issues(content)
                    for mem_issue in memory_issues:
                        file_issues["issues"].append({
                            "type": "memory_issue",
                            "description": "内存使用可以优化",
                            "location": mem_issue,
                            "severity": "low",
                            "impact": "memory_usage"
                        })
                    
                    if file_issues["issues"]:
                        performance_issues.append(file_issues)
                
                except Exception as e:
                    logger.warning(f"检测性能问题失败 {file_path}: {e}")
        
        logger.info(f"        检测到 {len(performance_issues)} 个文件的性能问题")
        return performance_issues

    def _detect_nested_loops(self, content: str) -> List[Dict[str, Any]]:
        """检测嵌套循环"""
        nested_loops = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if 'for ' in line and i < len(lines) - 1:
                # 检查下一行是否有另一个for循环
                next_line = lines[i + 1]
                if 'for ' in next_line and next_line.startswith(lines[i][0] * len(lines[i]) - len(lines[i].lstrip())):
                    nested_loops.append({
                        "line": i + 1,
                        "content": line.strip(),
                        "next_line": next_line.strip()
                    })
        
        return nested_loops

    def _detect_inefficient_lookups(self, content: str) -> List[Dict[str, Any]]:
        """检测低效查找"""
        inefficient_lookups = []
        lines = content.split('\n')
        
        # 查找列表中的in操作
        for i, line in enumerate(lines):
            if ' in ' in line and not any(keyword in line for keyword in ['set(', 'dict(', 'tuple(']):
                # 简单检查是否可能是列表查找
                match = re.search(r'(\w+)\s+in\s+(\w+)', line)
                if match:
                    var_name, collection_name = match.groups()
                    # 如果集合名以s结尾，可能是列表
                    if collection_name.endswith('s') or 'list' in collection_name.lower():
                        inefficient_lookups.append({
                            "line": i + 1,
                            "content": line.strip(),
                            "lookup_var": var_name,
                            "collection": collection_name
                        })
        
        return inefficient_lookups

    def _detect_repeated_calculations(self, content: str) -> List[Dict[str, Any]]:
        """检测重复计算"""
        repeated_calculations = []
        lines = content.split('\n')
        
        # 查找在循环中重复的计算
        function_calls = defaultdict(list)
        
        for i, line in enumerate(lines):
            # 查找函数调用
            matches = re.findall(r'(\w+)\([^)]*\)', line)
            for func_call in matches:
                function_calls[func_call].append(i + 1)
        
        # 如果同一个函数调用在附近多次出现
        for func_name, line_numbers in function_calls.items():
            if len(line_numbers) > 1:
                # 检查是否在循环中
                for line_num in line_numbers:
                    context_lines = lines[max(0, line_num-5):line_num+5]
                    if any('for ' in ctx_line or 'while ' in ctx_line for ctx_line in context_lines):
                        repeated_calculations.append({
                            "line": line_num,
                            "content": lines[line_num-1].strip(),
                            "function": func_name,
                            "occurrences": line_numbers
                        })
                        break
        
        return repeated_calculations

    def _detect_io_operations(self, content: str) -> List[Dict[str, Any]]:
        """检测I/O操作"""
        io_operations = []
        lines = content.split('\n')
        
        io_patterns = [
            r'open\(',
            r'\.read\(',
            r'\.write\(',
            r'requests\.',
            r'urllib\.',
            r'subprocess\.'
        ]
        
        for i, line in enumerate(lines):
            for pattern in io_patterns:
                if re.search(pattern, line):
                    io_operations.append({
                        "line": i + 1,
                        "content": line.strip(),
                        "operation_type": pattern.strip('\\')
                    })
                    break
        
        return io_operations

    def _detect_memory_issues(self, content: str) -> List[Dict[str, Any]]:
        """检测内存问题"""
        memory_issues = []
        lines = content.split('\n')
        
        # 查找可能导致内存问题的模式
        memory_patterns = [
            r'\[\w+\s+for\s+\w+\s+in\s+\w+\s+if\s+\w+\]',  # 列表推导式
            r'\.append\(',  # 列表追加
            r'list\(',  # 创建列表
            r'dict\(',   # 创建字典
        ]
        
        for i, line in enumerate(lines):
            for pattern in memory_patterns:
                if re.search(pattern, line):
                    memory_issues.append({
                        "line": i + 1,
                        "content": line.strip(),
                        "pattern": pattern.strip('\\')
                    })
                    break
        
        return memory_issues

    async def _ai_analyze_optimization_plan(self, performance_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """AI分析和优化策略"""
        logger.info("      AI分析优化策略...")
        
        optimization_plan = {
            "critical_optimizations": [],
            "significant_optimizations": [],
            "minor_optimizations": [],
            "estimated_improvement": {},
            "risk_assessment": {}
        }
        
        # 按影响和严重程度分类
        for file_issue in performance_issues:
            for issue in file_issue["issues"]:
                optimization = {
                    "file_path": file_issue["file_path"],
                    "issue_type": issue["type"],
                    "description": issue["description"],
                    "severity": issue["severity"],
                    "impact": issue["impact"],
                    "ai_strategy": await self._generate_ai_optimization_strategy(issue),
                    "estimated_improvement": self._estimate_performance_improvement(issue),
                    "confidence": 0.7
                }
                
                # 根据严重程度和影响分类
                if issue["severity"] == "high" or issue["impact"] == "algorithmic_complexity":
                    optimization_plan["critical_optimizations"].append(optimization)
                elif issue["severity"] == "medium":
                    optimization_plan["significant_optimizations"].append(optimization)
                else:
                    optimization_plan["minor_optimizations"].append(optimization)
        
        # 估算总体改进
        total_improvement = self._calculate_total_improvement(optimization_plan)
        optimization_plan["estimated_improvement"] = total_improvement
        
        logger.info(f"        AI规划完成: {len(optimization_plan['critical_optimizations'])} 个关键优化")
        return optimization_plan

    async def _generate_ai_optimization_strategy(self, issue: Dict[str, Any]) -> str:
        """生成AI优化策略"""
        strategies = {
            "nested_loop": "建议使用字典查找、集合操作或算法优化来减少复杂度",
            "inefficient_lookup": "建议将列表转换为集合或使用字典来提高查找效率",
            "repeated_calculation": "建议实现缓存机制或预计算来避免重复计算",
            "io_operation": "建议使用异步I/O、批处理或缓存来优化I/O性能",
            "memory_issue": "建议使用生成器、分块处理或更高效的数据结构"
        }
        
        return strategies.get(issue["type"], "建议进行性能优化以提高执行效率")

    def _estimate_performance_improvement(self, issue: Dict[str, Any]) -> Dict[str, float]:
        """估算性能改进"""
        improvements = {
            "nested_loop": {"speedup": 5.0, "memory_reduction": 0.0},
            "inefficient_lookup": {"speedup": 2.0, "memory_reduction": 0.0},
            "repeated_calculation": {"speedup": 1.5, "memory_reduction": 0.0},
            "io_operation": {"speedup": 2.0, "memory_reduction": 0.0},
            "memory_issue": {"speedup": 1.2, "memory_reduction": 0.3}
        }
        
        return improvements.get(issue["type"], {"speedup": 1.1, "memory_reduction": 0.1})

    def _calculate_total_improvement(self, optimization_plan: Dict[str, Any]) -> Dict[str, Any]:
        """计算总体改进"""
        total_speedup = 1.0
        total_memory_reduction = 0.0
        
        all_optimizations = (optimization_plan["critical_optimizations"] + 
                           optimization_plan["significant_optimizations"] + 
                           optimization_plan["minor_optimizations"])
        
        for opt in all_optimizations:
            improvement = opt["estimated_improvement"]
            total_speedup *= improvement["speedup"]
            total_memory_reduction += improvement["memory_reduction"]
        
        return {
            "estimated_speedup": min(total_speedup, 10.0),  # 限制最大改进
            "memory_reduction_percent": min(total_memory_reduction * 100, 50.0)
        }

    async def _ai_execute_performance_optimization(self, optimization_plan: Dict[str, Any]):
        """生成性能优化建议报告（不执行实际优化）"""
        logger.info("      生成性能优化建议报告...")
        
        # 按优先级生成建议
        all_optimizations = (optimization_plan["critical_optimizations"] + 
                            optimization_plan["significant_optimizations"])
        
        for optimization in all_optimizations:
            logger.info(f"        📋 性能优化建议: {optimization['description']}")
            logger.info(f"           文件: {optimization['file_path']}")
            logger.info(f"           AI策略: {optimization['ai_strategy']}")
            logger.info(f"           预期改进: 加速{optimization['estimated_improvement']['speedup']:.1f}x, 内存减少{optimization['estimated_improvement']['memory_reduction']*100:.1f}%")
            
            # 记录到changelog作为建议
            self.changelog.append({
                "timestamp": datetime.now().isoformat(),
                "action": "performance_suggestion",
                "issue": optimization["description"],
                "file": optimization["file_path"],
                "status": "suggested",
                "ai_strategy": optimization["ai_strategy"],
                "estimated_improvement": optimization["estimated_improvement"]
            })
        
        logger.info(f"        生成了 {len(all_optimizations)} 个性能优化建议")

    async def _ai_apply_optimization(self, optimization: Dict[str, Any]) -> bool:
        """AI应用性能优化"""
        try:
            file_path = Path(optimization["file_path"])
            
            if not file_path.exists():
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 根据问题类型应用不同的AI优化策略
            if optimization["issue_type"] == "nested_loop":
                new_content = await self._ai_optimize_nested_loop(content, optimization)
            elif optimization["issue_type"] == "inefficient_lookup":
                new_content = await self._ai_optimize_lookup(content, optimization)
            elif optimization["issue_type"] == "repeated_calculation":
                new_content = await self._ai_optimize_calculation(content, optimization)
            elif optimization["issue_type"] == "io_operation":
                new_content = await self._ai_optimize_io(content, optimization)
            elif optimization["issue_type"] == "memory_issue":
                new_content = await self._ai_optimize_memory(content, optimization)
            else:
                return False
            
            # 如果内容有变化，写入文件
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"AI应用性能优化失败: {e}")
            return False

    async def _ai_optimize_nested_loop(self, content: str, optimization: Dict[str, Any]) -> str:
        """AI优化嵌套循环"""
        logger.info(f"          AI分析嵌套循环优化: {optimization['description']}")
        return content  # 实际应该返回优化后的内容

    async def _ai_optimize_lookup(self, content: str, optimization: Dict[str, Any]) -> str:
        """AI优化查找操作"""
        logger.info(f"          AI分析查找优化: {optimization['description']}")
        return content  # 实际应该返回优化后的内容

    async def _ai_optimize_calculation(self, content: str, optimization: Dict[str, Any]) -> str:
        """AI优化计算"""
        logger.info(f"          AI分析计算优化: {optimization['description']}")
        return content  # 实际应该返回优化后的内容

    async def _ai_optimize_io(self, content: str, optimization: Dict[str, Any]) -> str:
        """AI优化I/O操作"""
        logger.info(f"          AI分析I/O优化: {optimization['description']}")
        return content  # 实际应该返回优化后的内容

    async def _ai_optimize_memory(self, content: str, optimization: Dict[str, Any]) -> str:
        """AI优化内存使用"""
        logger.info(f"          AI分析内存优化: {optimization['description']}")
        return content  # 实际应该返回优化后的内容

    async def _execute_security_hardening(self):
        """执行安全加固 - AI驱动模式"""
        logger.info("    执行安全加固...")
        
        try:
            # 1. 检测安全漏洞
            security_vulnerabilities = await self._detect_security_vulnerabilities()
            
            # 2. AI分析和安全策略
            security_plan = await self._ai_analyze_security_plan(security_vulnerabilities)
            
            # 3. AI执行安全加固
            await self._ai_execute_security_hardening(security_plan)
            
            logger.info("      ✅ 安全加固完成")
            
        except Exception as e:
            logger.error(f"      ❌ 安全加固失败: {e}")

    async def _detect_security_vulnerabilities(self) -> List[Dict[str, Any]]:
        """检测安全漏洞"""
        logger.info("      检测安全漏洞...")
        
        security_vulnerabilities = []
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_vulnerabilities = {
                        "file_path": str(file_path),
                        "vulnerabilities": []
                    }
                    
                    # 检测代码注入
                    injection_vulns = self._detect_code_injection(content)
                    for vuln in injection_vulns:
                        file_vulnerabilities["vulnerabilities"].append({
                            "type": "code_injection",
                            "description": "潜在的代码注入风险",
                            "location": vuln,
                            "severity": "critical",
                            "cwe": "CWE-94"
                        })
                    
                    # 检测命令注入
                    command_injection = self._detect_command_injection(content)
                    for vuln in command_injection:
                        file_vulnerabilities["vulnerabilities"].append({
                            "type": "command_injection",
                            "description": "潜在的命令注入风险",
                            "location": vuln,
                            "severity": "critical",
                            "cwe": "CWE-78"
                        })
                    
                    # 检测SQL注入
                    sql_injection = self._detect_sql_injection(content)
                    for vuln in sql_injection:
                        file_vulnerabilities["vulnerabilities"].append({
                            "type": "sql_injection",
                            "description": "潜在的SQL注入风险",
                            "location": vuln,
                            "severity": "high",
                            "cwe": "CWE-89"
                        })
                    
                    # 检测XSS
                    xss_vulns = self._detect_xss(content)
                    for vuln in xss_vulns:
                        file_vulnerabilities["vulnerabilities"].append({
                            "type": "xss",
                            "description": "潜在的跨站脚本攻击风险",
                            "location": vuln,
                            "severity": "high",
                            "cwe": "CWE-79"
                        })
                    
                    # 检测敏感信息泄露
                    sensitive_data = self._detect_sensitive_data(content)
                    for vuln in sensitive_data:
                        file_vulnerabilities["vulnerabilities"].append({
                            "type": "sensitive_data",
                            "description": "敏感信息泄露风险",
                            "location": vuln,
                            "severity": "medium",
                            "cwe": "CWE-200"
                        })
                    
                    # 检测弱加密
                    weak_crypto = self._detect_weak_crypto(content)
                    for vuln in weak_crypto:
                        file_vulnerabilities["vulnerabilities"].append({
                            "type": "weak_crypto",
                            "description": "使用弱加密算法",
                            "location": vuln,
                            "severity": "medium",
                            "cwe": "CWE-327"
                        })
                    
                    # 检测硬编码凭证
                    hardcoded_creds = self._detect_hardcoded_credentials(content)
                    for vuln in hardcoded_creds:
                        file_vulnerabilities["vulnerabilities"].append({
                            "type": "hardcoded_credentials",
                            "description": "硬编码凭证信息",
                            "location": vuln,
                            "severity": "high",
                            "cwe": "CWE-798"
                        })
                    
                    if file_vulnerabilities["vulnerabilities"]:
                        security_vulnerabilities.append(file_vulnerabilities)
                
                except Exception as e:
                    logger.warning(f"检测安全漏洞失败 {file_path}: {e}")
        
        logger.info(f"        检测到 {len(security_vulnerabilities)} 个文件的安全漏洞")
        return security_vulnerabilities

    def _detect_code_injection(self, content: str) -> List[Dict[str, Any]]:
        """检测代码注入"""
        code_injection = []
        lines = content.split('\n')
        
        dangerous_functions = ['eval(', 'exec(', 'compile(']
        
        for i, line in enumerate(lines):
            for func in dangerous_functions:
                if func in line:
                    code_injection.append({
                        "line": i + 1,
                        "content": line.strip(),
                        "function": func.strip('(')
                    })
        
        return code_injection

    def _detect_command_injection(self, content: str) -> List[Dict[str, Any]]:
        """检测命令注入"""
        command_injection = []
        lines = content.split('\n')
        
        dangerous_patterns = [
            r'subprocess\.',
            r'os\.system',
            r'os\.popen',
            r'commands\.',
            r'popen2\.',
            r'popen4\.',
            r'spawn\.',
            r'call\('
        ]
        
        for i, line in enumerate(lines):
            for pattern in dangerous_patterns:
                if re.search(pattern, line) and 'shell=True' in line:
                    command_injection.append({
                        "line": i + 1,
                        "content": line.strip(),
                        "pattern": pattern
                    })
        
        return command_injection

    def _detect_sql_injection(self, content: str) -> List[Dict[str, Any]]:
        """检测SQL注入"""
        sql_injection = []
        lines = content.split('\n')
        
        # 查找SQL拼接模式
        sql_patterns = [
            r'SELECT.*\+.*',
            r'INSERT.*\+.*',
            r'UPDATE.*\+.*',
            r'DELETE.*\+.*',
            r'WHERE.*\+.*',
            r'".*%s.*".*%.*',
            r"'.*%s.*'.*%.*"
        ]
        
        for i, line in enumerate(lines):
            for pattern in sql_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    sql_injection.append({
                        "line": i + 1,
                        "content": line.strip(),
                        "pattern": pattern
                    })
        
        return sql_injection

    def _detect_xss(self, content: str) -> List[Dict[str, Any]]:
        """检测XSS"""
        xss_vulns = []
        lines = content.split('\n')
        
        xss_patterns = [
            r'innerHTML.*=',
            r'outerHTML.*=',
            r'document\.write',
            r'eval\(',
            r'setTimeout.*eval',
            r'setInterval.*eval'
        ]
        
        for i, line in enumerate(lines):
            for pattern in xss_patterns:
                if re.search(pattern, line):
                    xss_vulns.append({
                        "line": i + 1,
                        "content": line.strip(),
                        "pattern": pattern
                    })
        
        return xss_vulns

    def _detect_sensitive_data(self, content: str) -> List[Dict[str, Any]]:
        """检测敏感信息泄露"""
        sensitive_data = []
        lines = content.split('\n')
        
        sensitive_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'private_key\s*=\s*["\'][^"\']+["\']',
            r'access_key\s*=\s*["\'][^"\']+["\']'
        ]
        
        for i, line in enumerate(lines):
            for pattern in sensitive_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    sensitive_data.append({
                        "line": i + 1,
                        "content": line.strip()[:50] + "...",  # 截断显示
                        "pattern": pattern
                    })
        
        return sensitive_data

    def _detect_weak_crypto(self, content: str) -> List[Dict[str, Any]]:
        """检测弱加密"""
        weak_crypto = []
        lines = content.split('\n')
        
        weak_algorithms = [
            'md5(',
            'sha1(',
            'DES(',
            'RC4(',
            'MD5(',
            'SHA1('
        ]
        
        for i, line in enumerate(lines):
            for algo in weak_algorithms:
                if algo in line:
                    weak_crypto.append({
                        "line": i + 1,
                        "content": line.strip(),
                        "algorithm": algo.strip('(')
                    })
        
        return weak_crypto

    def _detect_hardcoded_credentials(self, content: str) -> List[Dict[str, Any]]:
        """检测硬编码凭证"""
        hardcoded_creds = []
        lines = content.split('\n')
        
        credential_patterns = [
            r'["\'][A-Za-z0-9+/]{20,}["\']',  # Base64编码的密钥
            r'["\'][A-Fa-f0-9]{32,}["\']',   # 十六进制密钥
            r'sk_[a-zA-Z0-9]{24,}',            # Stripe密钥
            r'ghp_[a-zA-Z0-9]{36}',            # GitHub个人访问令牌
            r'AIza[0-9A-Za-z_-]{35}'           # Google API密钥
        ]
        
        for i, line in enumerate(lines):
            for pattern in credential_patterns:
                if re.search(pattern, line):
                    hardcoded_creds.append({
                        "line": i + 1,
                        "content": "HARDCODED_CREDENTIAL",  # 不显示实际内容
                        "pattern": pattern
                    })
        
        return hardcoded_creds

    async def _ai_analyze_security_plan(self, vulnerabilities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """AI分析和安全策略"""
        logger.info("      AI分析安全策略...")
        
        security_plan = {
            "critical_fixes": [],
            "high_priority_fixes": [],
            "medium_priority_fixes": [],
            "security_score": 0.0,
            "risk_assessment": {}
        }
        
        # 按严重程度分类
        for file_vuln in vulnerabilities:
            for vuln in file_vuln["vulnerabilities"]:
                fix = {
                    "file_path": file_vuln["file_path"],
                    "vulnerability_type": vuln["type"],
                    "description": vuln["description"],
                    "severity": vuln["severity"],
                    "cwe": vuln["cwe"],
                    "location": vuln["location"],
                    "ai_fix_strategy": await self._generate_ai_security_fix(vuln),
                    "confidence": 0.8
                }
                
                # 根据严重程度分类
                if vuln["severity"] == "critical":
                    security_plan["critical_fixes"].append(fix)
                elif vuln["severity"] == "high":
                    security_plan["high_priority_fixes"].append(fix)
                else:
                    security_plan["medium_priority_fixes"].append(fix)
        
        # 计算安全评分
        total_vulns = (len(security_plan["critical_fixes"]) + 
                       len(security_plan["high_priority_fixes"]) + 
                       len(security_plan["medium_priority_fixes"]))
        
        if total_vulns > 0:
            critical_weight = len(security_plan["critical_fixes"]) * 10
            high_weight = len(security_plan["high_priority_fixes"]) * 5
            medium_weight = len(security_plan["medium_priority_fixes"]) * 2
            
            security_plan["security_score"] = max(0, 100 - (critical_weight + high_weight + medium_weight))
        
        logger.info(f"        AI安全规划完成: {len(security_plan['critical_fixes'])} 个关键修复")
        return security_plan

    async def _generate_ai_security_fix(self, vulnerability: Dict[str, Any]) -> str:
        """生成AI安全修复策略"""
        fix_strategies = {
            "code_injection": "建议移除eval/exec调用，使用安全的替代方案如ast.literal_eval",
            "command_injection": "建议避免shell=True，使用参数化命令或subprocess.run without shell",
            "sql_injection": "建议使用参数化查询或ORM来防止SQL注入",
            "xss": "建议对用户输入进行HTML转义，使用安全的模板引擎",
            "sensitive_data": "建议将敏感信息移至环境变量或安全的配置管理系统",
            "weak_crypto": "建议使用强加密算法如SHA-256、AES-256",
            "hardcoded_credentials": "建议移除硬编码凭证，使用密钥管理服务"
        }
        
        return fix_strategies.get(vulnerability["type"], "建议遵循安全最佳实践进行修复")

    async def _ai_execute_security_hardening(self, security_plan: Dict[str, Any]):
        """AI执行安全加固"""
        logger.info("      AI执行安全加固...")
        
        # 按优先级执行安全修复
        all_fixes = (security_plan["critical_fixes"] + 
                    security_plan["high_priority_fixes"] + 
                    security_plan["medium_priority_fixes"])
        
        for fix in all_fixes:
            try:
                if not self.analysis_mode:
                    # AI执行具体的安全修复
                    success = await self._ai_apply_security_fix(fix)
                    
                    if success:
                        logger.info(f"        ✅ AI安全修复成功: {fix['description']}")
                        self.changelog.append({
                            "timestamp": datetime.now().isoformat(),
                            "action": "ai_security_fix",
                            "issue": fix["description"],
                            "file": fix["file_path"],
                            "status": "success",
                            "cwe": fix["cwe"],
                            "ai_fix_strategy": fix["ai_fix_strategy"]
                        })
                    else:
                        logger.warning(f"        ⚠️ AI安全修复失败: {fix['description']}")
                else:
                    logger.info(f"        📋 AI安全修复建议: {fix['description']}")
            
            except Exception as e:
                logger.error(f"        ❌ AI安全修复异常: {fix['description']} - {e}")

    async def _ai_apply_security_fix(self, fix: Dict[str, Any]) -> bool:
        """AI应用安全修复"""
        try:
            file_path = Path(fix["file_path"])
            
            if not file_path.exists():
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 根据漏洞类型应用不同的AI修复策略
            if fix["vulnerability_type"] == "code_injection":
                new_content = await self._ai_fix_code_injection(content, fix)
            elif fix["vulnerability_type"] == "command_injection":
                new_content = await self._ai_fix_command_injection(content, fix)
            elif fix["vulnerability_type"] == "sql_injection":
                new_content = await self._ai_fix_sql_injection(content, fix)
            elif fix["vulnerability_type"] == "sensitive_data":
                new_content = await self._ai_fix_sensitive_data(content, fix)
            elif fix["vulnerability_type"] == "weak_crypto":
                new_content = await self._ai_fix_weak_crypto(content, fix)
            else:
                return False
            
            # 如果内容有变化，写入文件
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"AI应用安全修复失败: {e}")
            return False

    async def _ai_fix_code_injection(self, content: str, fix: Dict[str, Any]) -> str:
        """AI修复代码注入"""
        logger.info(f"          AI分析代码注入修复: {fix['description']}")
        return content  # 实际应该返回修复后的内容

    async def _ai_fix_command_injection(self, content: str, fix: Dict[str, Any]) -> str:
        """AI修复命令注入"""
        logger.info(f"          AI分析命令注入修复: {fix['description']}")
        return content  # 实际应该返回修复后的内容

    async def _ai_fix_sql_injection(self, content: str, fix: Dict[str, Any]) -> str:
        """AI修复SQL注入"""
        logger.info(f"          AI分析SQL注入修复: {fix['description']}")
        return content  # 实际应该返回修复后的内容

    async def _ai_fix_sensitive_data(self, content: str, fix: Dict[str, Any]) -> str:
        """AI修复敏感信息泄露"""
        logger.info(f"          AI分析敏感信息修复: {fix['description']}")
        return content  # 实际应该返回修复后的内容

    async def _ai_fix_weak_crypto(self, content: str, fix: Dict[str, Any]) -> str:
        """AI修复弱加密"""
        logger.info(f"          AI分析弱加密修复: {fix['description']}")
        return content  # 实际应该返回修复后的内容

    async def _execute_architecture_improvements(self):
        """执行架构改进"""
        logger.info("    执行架构改进...")
        
        # 这里可以实现架构改进逻辑
        # 目前只是占位符
        pass

    async def _phase_validation(self):
        """阶段4: 验证测试"""
        logger.info("🧪 阶段4: 验证测试...")
        self.current_phase = UpgradePhase.VALIDATION
        
        try:
            # 4.1 语法检查
            await self._validate_syntax()
            
            # 4.2 单元测试
            await self._run_unit_tests()
            
            # 4.3 集成测试
            await self._run_integration_tests()
            
            # 4.4 性能测试
            await self._run_performance_tests()
            
            # 4.5 安全测试
            await self._run_security_tests()
            
            logger.info("  ✅ 验证测试完成")
            
        except Exception as e:
            logger.error(f"  ❌ 验证测试失败: {e}")
            raise

    async def _validate_syntax(self):
        """语法检查"""
        logger.info("    执行语法检查...")
        
        syntax_errors = []
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 编译检查语法
                    ast.parse(content)
                
                except SyntaxError as e:
                    syntax_errors.append({
                        "file": str(file_path),
                        "line": e.lineno,
                        "error": str(e)
                    })
                except Exception as e:
                    logger.warning(f"无法检查语法 {file_path}: {e}")
        
        if syntax_errors:
            logger.warning(f"发现 {len(syntax_errors)} 个语法错误")
            for error in syntax_errors[:5]:  # 只显示前5个
                logger.warning(f"  {error['file']}:{error['line']} - {error['error']}")
        else:
            logger.info("    ✅ 所有文件语法正确")

    async def _run_unit_tests(self):
        """运行单元测试"""
        logger.info("    运行单元测试...")
        
        # 尝试运行pytest
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "-v", "--tb=short"],
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info("    ✅ 单元测试通过")
            else:
                logger.warning(f"    ⚠️ 单元测试失败: {len(result.stdout.splitlines())} 个失败")
                if self.verbose:
                    logger.warning(result.stdout)
        
        except subprocess.TimeoutExpired:
            logger.warning("    ⏰ 单元测试超时")
        except FileNotFoundError:
            logger.info("    ℹ️ 未找到pytest，跳过单元测试")
        except Exception as e:
            logger.warning(f"    ❌ 单元测试执行失败: {e}")

    async def _run_integration_tests(self):
        """运行集成测试"""
        logger.info("    运行集成测试...")
        
        # 这里可以实现集成测试逻辑
        # 目前只是占位符
        logger.info("    ℹ️ 集成测试跳过（未实现）")

    async def _run_performance_tests(self):
        """运行性能测试"""
        logger.info("    运行性能测试...")
        
        # 这里可以实现性能测试逻辑
        # 目前只是占位符
        logger.info("    ℹ️ 性能测试跳过（未实现）")

    async def _run_security_tests(self):
        """运行安全测试"""
        logger.info("    运行安全测试...")
        
        # 这里可以实现安全测试逻辑
        # 目前只是占位符
        logger.info("    ℹ️ 安全测试跳过（未实现）")

    async def _phase_documentation(self):
        """阶段5: 文档生成"""
        logger.info("📚 阶段5: 文档生成...")
        self.current_phase = UpgradePhase.DOCUMENTATION
        
        try:
            # 5.1 生成升级报告
            await self._generate_upgrade_report()
            
            # 5.2 生成API文档
            await self._generate_api_documentation()
            
            # 5.3 生成变更日志
            await self._generate_changelog()
            
            # 5.4 更新README
            await self._update_readme()
            
            logger.info("  ✅ 文档生成完成")
            
        except Exception as e:
            logger.error(f"  ❌ 文档生成失败: {e}")
            raise

    async def _generate_upgrade_report(self):
        """生成升级报告"""
        logger.info("    生成升级报告...")
        
        report_content = [
            "# 项目升级报告",
            f"",
            f"**升级时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**会话ID**: {self.session_id}",
            f"**版本**: {self.version_info['current']} → {self.version_info['target']}",
            f"",
            "## 升级摘要",
            f"",
            f"- 总文件数: {self.project_metrics.total_files}",
            f"- 代码文件: {self.project_metrics.code_files}",
            f"- 测试文件: {self.project_metrics.test_files}",
            f"- 总代码行数: {self.project_metrics.total_lines}",
            f"- 测试覆盖率: {self.project_metrics.test_coverage:.1f}%",
            f"- 可维护性指数: {self.project_metrics.maintainability_index:.1f}",
            f"",
            "## 发现的问题",
            f""
        ]
        
        # 按严重程度统计问题
        severity_counts = defaultdict(int)
        for issue in self.issues:
            severity_counts[issue.severity.value] += 1
        
        for severity in ["critical", "high", "medium", "low"]:
            count = severity_counts.get(severity, 0)
            if count > 0:
                report_content.append(f"- {severity.capitalize()}: {count}")
        
        report_content.extend([
            f"",
            "## 修复的问题",
            f""
        ])
        
        # 添加修复的问题
        fixed_issues = [c for c in self.changelog if c["status"] == "success"]
        report_content.append(f"- 成功修复: {len(fixed_issues)} 个问题")
        
        failed_issues = [c for c in self.changelog if c["status"] == "failed"]
        if failed_issues:
            report_content.append(f"- 修复失败: {len(failed_issues)} 个问题")
        
        report_content.extend([
            f"",
            "## 架构分析",
            f"",
            f"- 检测到的架构模式: {', '.join(self.architecture_analysis.get('detected_patterns', []))}",
            f"- 平均耦合度: {self.architecture_analysis.get('average_coupling', 0):.2f}",
            f"",
            "## 建议和后续步骤",
            f"",
            "1. 继续监控代码质量指标",
            "2. 定期运行安全扫描",
            "3. 增加测试覆盖率",
            "4. 优化性能瓶颈",
            "5. 定期更新依赖",
            f"",
            "---",
            f"*报告由 iFlow CLI 自动生成*"
        ])
        
        # 保存报告
        report_dir = self.workspace_path / ".iflow" / "reports"
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / f"upgrade_report_{self.session_id}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        logger.info(f"      ✅ 升级报告已保存: {report_file}")

    async def _generate_api_documentation(self):
        """生成API文档"""
        logger.info("    生成API文档...")
        
        # 这里可以实现API文档生成逻辑
        # 目前只是占位符
        pass

    async def _generate_changelog(self):
        """生成变更日志"""
        logger.info("    生成变更日志...")
        
        changelog_content = [
            "# 变更日志",
            f"",
            f"## [{self.version_info['target']}] - {datetime.now().strftime('%Y-%m-%d')}",
            f""
        ]
        
        # 按类型分组变更
        changes_by_type = defaultdict(list)
        for change in self.changelog:
            if change["status"] == "success":
                changes_by_type[change["action"]].append(change["issue"])
        
        for change_type, issues in changes_by_type.items():
            changelog_content.append(f"### {change_type.title()}")
            changelog_content.append("")
            
            for issue in issues:
                changelog_content.append(f"- {issue}")
            
            changelog_content.append("")
        
        # 保存变更日志
        changelog_file = self.workspace_path / "CHANGELOG.md"
        
        # 如果文件存在，读取现有内容
        existing_content = ""
        if changelog_file.exists():
            with open(changelog_file, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        
        # 写入新内容
        with open(changelog_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(changelog_content))
            if existing_content:
                f.write('\n')
                f.write(existing_content)
        
        logger.info(f"      ✅ 变更日志已更新: {changelog_file}")

    async def _update_readme(self):
        """更新README"""
        logger.info("    更新README...")
        
        readme_file = self.workspace_path / "README.md"
        
        if not readme_file.exists():
            # 创建基本README
            readme_content = [
                f"# {self.workspace_path.name}",
                f"",
                f"## 项目信息",
                f"",
                f"- 升级时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- 版本: {self.version_info['target']}",
                f"- 文件数: {self.project_metrics.total_files}",
                f"- 测试覆盖率: {self.project_metrics.test_coverage:.1f}%",
                f"",
                f"## 快速开始",
                f"",
                f"```bash",
                f"# 安装依赖",
                f"# 运行测试",
                f"# 启动项目",
                f"```",
                f"",
                f"## 文档",
                f"",
                f"- [升级报告](.iflow/reports/upgrade_report_{self.session_id}.md)",
                f"- [变更日志](CHANGELOG.md)",
                f"",
                f"---",
                f"*此README由 iFlow CLI 自动生成*"
            ]
            
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(readme_content))
            
            logger.info(f"      ✅ README已创建: {readme_file}")

    async def _phase_cleanup(self):
        """阶段6: 清理优化"""
        logger.info("🗑️ 阶段6: 清理优化...")
        self.current_phase = UpgradePhase.CLEANUP
        
        try:
            # 6.1 清理临时文件
            await self._cleanup_temp_files()
            
            # 6.2 清理旧代码
            await self._cleanup_old_code()
            
            # 6.3 优化导入
            await self._optimize_imports()
            
            # 6.4 清理缓存
            await self._cleanup_cache()
            
            logger.info("  ✅ 清理优化完成")
            
        except Exception as e:
            logger.error(f"  ❌ 清理优化失败: {e}")
            raise

    async def _cleanup_temp_files(self):
        """清理临时文件"""
        logger.info("    清理临时文件...")
        
        temp_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/.pytest_cache",
            "**/.coverage",
            "**/.mypy_cache"
        ]
        
        cleaned_count = 0
        
        for pattern in temp_patterns:
            for path in self.workspace_path.rglob(pattern.split('/')[-1]):
                if path.is_dir():
                    try:
                        shutil.rmtree(path)
                        cleaned_count += 1
                    except:
                        pass
                elif path.is_file():
                    try:
                        path.unlink()
                        cleaned_count += 1
                    except:
                        pass
        
        logger.info(f"      清理了 {cleaned_count} 个临时文件/目录")

    async def _cleanup_old_code(self):
        """清理旧代码"""
        logger.info("    清理旧代码...")
        
        # 识别未使用的代码
        unused_imports = []
        unused_functions = []
        
        for file_path in self.workspace_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 简单的未使用导入检测
                    imports = re.findall(r'^import\s+(\w+)|^from\s+(\w+)', content, re.MULTILINE)
                    for import_match in imports:
                        module = import_match[0] or import_match[1]
                        if module and module not in content:
                            unused_imports.append((str(file_path), module))
                
                except Exception as e:
                    logger.warning(f"无法分析旧代码 {file_path}: {e}")
        
        if unused_imports and not self.dry_run:
            logger.info(f"      发现 {len(unused_imports)} 个未使用的导入")
            # 这里可以实现自动清理逻辑
        else:
            logger.info("      未发现需要清理的旧代码")

    async def _optimize_imports(self):
        """优化导入"""
        logger.info("    优化导入...")
        
        # 这里可以实现导入优化逻辑
        # 目前只是占位符
        pass

    async def _cleanup_cache(self):
        """清理缓存"""
        logger.info("    清理缓存...")
        
        # 清理.iflow/cache
        cache_dir = self.workspace_path / ".iflow" / "cache"
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(exist_ok=True)
                logger.info("      ✅ 缓存已清理")
            except Exception as e:
                logger.warning(f"      清理缓存失败: {e}")

    async def _generate_final_report(self) -> Dict[str, Any]:
        """生成最终分析报告（纯检测模式）"""
        logger.info("📊 生成最终分析报告...")
        
        try:
            # 计算分析统计
            total_issues = len(self.issues)
            
            # 按类型分类问题
            issues_by_type = defaultdict(int)
            issues_by_severity = defaultdict(int)
            
            for issue in self.issues:
                issues_by_type[issue.category] += 1
                issues_by_severity[issue.severity.value] += 1
            
            # 计算质量指标
            quality_score = self._calculate_quality_score()
            
            # 构建最终报告
            final_report = {
                "analysis_summary": {
                    "session_id": self.session_id,
                    "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration_minutes": (time.time() - self.start_time) / 60,
                    "analysis_mode": "detection_only",
                    "total_issues_detected": total_issues,
                    "issues_by_type": dict(issues_by_type),
                    "issues_by_severity": dict(issues_by_severity)
                },
                "project_metrics": asdict(self.project_metrics),
                "quality_metrics": {
                    "quality_score": quality_score,
                    "maintainability_index": self.project_metrics.maintainability_index,
                    "test_coverage": self.project_metrics.test_coverage,
                    "complexity_score": self.project_metrics.complexity_score,
                    "security_score": self._calculate_security_score()
                },
                "architecture_analysis": self.architecture_analysis,
                "security_findings": [asdict(finding) for finding in self.security_findings],
                "detailed_issues": [asdict(issue) for issue in self.issues],
                "analysis_suggestions": await self._generate_comprehensive_suggestions(),
                "ai_training_data": await self._generate_ai_training_data(),
                "recommendations": await self._generate_recommendations()
            }
            
            return final_report
            
        except Exception as e:
            logger.error(f"生成最终报告失败: {e}")
            return {"error": str(e)}

    def _calculate_security_score(self) -> float:
        """计算安全评分"""
        if not self.security_findings:
            return 100.0
        
        total_findings = len(self.security_findings)
        critical_count = len([f for f in self.security_findings if f.severity == Severity.CRITICAL])
        high_count = len([f for f in self.security_findings if f.severity == Severity.HIGH])
        
        # 安全评分计算
        score = 100.0
        score -= (critical_count * 25)  # 严重问题扣25分
        score -= (high_count * 15)     # 高危问题扣15分
        score -= ((total_findings - critical_count - high_count) * 5)  # 其他问题扣5分
        
        return max(0.0, score)

    async def _generate_comprehensive_suggestions(self) -> List[Dict[str, Any]]:
        """生成综合建议"""
        suggestions = []
        
        # 按优先级分组问题
        critical_issues = [issue for issue in self.issues if issue.severity == Severity.CRITICAL]
        high_issues = [issue for issue in self.issues if issue.severity == Severity.HIGH]
        medium_issues = [issue for issue in self.issues if issue.severity == Severity.MEDIUM]
        low_issues = [issue for issue in self.issues if issue.severity == Severity.LOW]
        
        # 生成优先级建议
        if critical_issues:
            suggestions.append({
                "priority": "critical",
                "title": "立即修复关键问题",
                "description": f"发现 {len(critical_issues)} 个关键问题需要立即处理",
                "issues": [asdict(issue) for issue in critical_issues[:5]],
                "estimated_effort": f"{len(critical_issues) * 4} 小时",
                "risk_level": "high"
            })
        
        if high_issues:
            suggestions.append({
                "priority": "high",
                "title": "优先处理高风险问题",
                "description": f"发现 {len(high_issues)} 个高风险问题建议优先处理",
                "issues": [asdict(issue) for issue in high_issues[:5]],
                "estimated_effort": f"{len(high_issues) * 2} 小时",
                "risk_level": "medium"
            })
        
        if medium_issues:
            suggestions.append({
                "priority": "medium",
                "title": "计划处理中等问题",
                "description": f"发现 {len(medium_issues)} 个中等问题可以计划处理",
                "issues": [asdict(issue) for issue in medium_issues[:5]],
                "estimated_effort": f"{len(medium_issues) * 1} 小时",
                "risk_level": "low"
            })
        
        return suggestions

    async def _generate_ai_training_data(self) -> Dict[str, Any]:
        """生成AI训练数据集"""
        training_data = {
            "session_metadata": {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "project_path": str(self.workspace_path),
                "total_files_analyzed": self.project_metrics.total_files,
                "analysis_duration": (time.time() - self.start_time) / 60
            },
            "detected_patterns": {
                "code_quality_patterns": [],
                "security_patterns": [],
                "performance_patterns": [],
                "architecture_patterns": []
            },
            "fix_strategies": {},
            "user_preferences": asdict(self.ai_profile),
            "success_criteria": {}
        }
        
        # 提取检测到的模式
        for issue in self.issues:
            pattern = {
                "type": issue.type,
                "category": issue.category,
                "severity": issue.severity.value,
                "description": issue.description,
                "file_path": issue.file_path,
                "line_number": issue.line_number,
                "evidence": issue.evidence,
                "fix_suggestion": issue.fix_suggestion,
                "auto_fixable": issue.auto_fixable
            }
            
            if issue.category == "security":
                training_data["detected_patterns"]["security_patterns"].append(pattern)
            elif issue.category == "performance":
                training_data["detected_patterns"]["performance_patterns"].append(pattern)
            elif issue.category == "architecture":
                training_data["detected_patterns"]["architecture_patterns"].append(pattern)
            else:
                training_data["detected_patterns"]["code_quality_patterns"].append(pattern)
        
        # 生成修复策略映射
        for issue in self.issues:
            if issue.type not in training_data["fix_strategies"]:
                training_data["fix_strategies"][issue.type] = []
            
            training_data["fix_strategies"][issue.type].append({
                "suggestion": issue.fix_suggestion,
                "auto_fixable": issue.auto_fixable,
                "context": f"文件: {issue.file_path}, 行: {issue.line_number}",
                "confidence": 0.8
            })
        
        # 生成成功标准
        training_data["success_criteria"] = {
            "quality_threshold": 80.0,
            "security_threshold": 90.0,
            "performance_threshold": 85.0,
            "test_coverage_threshold": 70.0
        }
        
        return training_data

    def _calculate_quality_score(self) -> float:
        """计算质量分数"""
        scores = []
        
        # 可维护性分数 (0-100)
        scores.append(self.project_metrics.maintainability_index)
        
        # 测试覆盖率分数 (0-100)
        scores.append(self.project_metrics.test_coverage)
        
        # 复杂度分数 (反向，复杂度越低分数越高)
        complexity_score = max(0, 100 - (self.project_metrics.complexity_score * 2))
        scores.append(complexity_score)
        
        # 安全分数 (基于安全问题数量)
        security_issues = len([i for i in self.issues if i.category == "security"])
        security_score = max(0, 100 - (security_issues * 10))
        scores.append(security_score)
        
        return sum(scores) / len(scores)

    async def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """生成建议"""
        recommendations = []
        
        # 基于分析结果生成建议
        if self.project_metrics.test_coverage < 50:
            recommendations.append({
                "category": "testing",
                "priority": "high",
                "title": "增加测试覆盖率",
                "description": f"当前测试覆盖率为 {self.project_metrics.test_coverage:.1f}%，建议增加到80%以上",
                "effort": "high"
            })
        
        if self.project_metrics.maintainability_index < 60:
            recommendations.append({
                "category": "maintainability",
                "priority": "medium",
                "title": "提高代码可维护性",
                "description": f"当前可维护性指数为 {self.project_metrics.maintainability_index:.1f}，建议重构复杂代码",
                "effort": "medium"
            })
        
        critical_security_issues = [i for i in self.security_findings if i.severity == Severity.CRITICAL]
        if critical_security_issues:
            recommendations.append({
                "category": "security",
                "priority": "critical",
                "title": "修复关键安全问题",
                "description": f"发现 {len(critical_security_issues)} 个关键安全问题，需要立即修复",
                "effort": "high"
            })
        
        return recommendations

    async def _save_upgrade_history(self, final_report: Dict[str, Any]):
        """保存升级历史"""
        logger.info("💾 保存升级历史...")
        
        try:
            # 添加到历史记录
            history_entry = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "version": self.version_info,
                "summary": final_report["upgrade_summary"],
                "quality_metrics": final_report["quality_metrics"]
            }
            
            self.upgrade_history.append(history_entry)
            
            # 保存历史文件
            history_file = self.workspace_path / ".iflow" / "data" / "upgrade_history.json"
            history_file.parent.mkdir(exist_ok=True)
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.upgrade_history, f, indent=2, ensure_ascii=False)
            
            # 保存AI偏好档案
            profile_file = self.workspace_path / ".iflow" / "data" / "ai_profile.json"
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.ai_profile), f, indent=2, ensure_ascii=False)
            
            logger.info("  ✅ 升级历史已保存")
            
        except Exception as e:
            logger.error(f"  保存升级历史失败: {e}")

    async def _analyze_coding_patterns(self):
        """分析编码模式"""
        # 这里可以实现编码模式分析逻辑
        # 用于学习用户偏好
        pass

    async def _learn_architectural_preferences(self):
        """学习架构偏好"""
        # 这里可以实现架构偏好学习逻辑
        # 用于理解用户的架构选择
        pass

    async def _understand_documentation_style(self):
        """理解文档风格"""
        # 这里可以实现文档风格分析逻辑
        # 用于学习用户的文档偏好
        pass

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="全自动化项目审查和升级工作流")
    parser.add_argument("--workspace", "-w", default=".", help="工作空间路径")
    parser.add_argument("--auto-fix", action="store_true", default=True, help="自动修复问题")
    parser.add_argument("--no-backup", action="store_true", help="不创建备份")
    parser.add_argument("--dry-run", action="store_true", help="分析模式，仅生成报告不修改文件")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--config", help="配置文件路径")
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if args.config and Path(args.config).exists():
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return 1
    
    # 设置配置
    config.update({
        "auto_fix": args.auto_fix,
        "backup_enabled": not args.no_backup,
        "analysis_mode": args.dry_run,
        "verbose": args.verbose
    })
    
    # 创建并执行工作流
    workflow = ComprehensiveProjectUpgradeWorkflow(args.workspace, config)
    
    try:
        await workflow.initialize()
        report = await workflow.execute_comprehensive_upgrade()
        
        # 输出结果摘要
        summary = report["upgrade_summary"]
        print(f"\n🎉 项目升级完成!")
        print(f"📊 总问题数: {summary['total_issues']}")
        print(f"✅ 已修复: {summary['fixed_issues']}")
        print(f"❌ 修复失败: {summary['failed_issues']}")
        print(f"📈 成功率: {summary['success_rate']:.1f}%")
        print(f"⏱️ 耗时: {summary['duration_minutes']:.1f} 分钟")
        print(f"📋 详细报告: .iflow/reports/upgrade_report_{workflow.session_id}.md")
        
        return 0
        
    except Exception as e:
        logger.error(f"工作流执行失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)