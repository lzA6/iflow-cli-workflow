#!/usr/bin/env python3
"""
项目结构树对比分析模块
提供详细的项目结构变化分析和决策支持
"""

import os
import json
import hashlib
import difflib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import ast
import re

@dataclass
class FileAnalysis:
    """文件分析结果"""
    path: str
    name: str
    size: int
    modified_time: float
    file_hash: str
    file_type: str
    functions: List[str]
    classes: List[str]
    imports: List[str]
    complexity_score: float
    dependencies: List[str]
    functionality_score: float
    maintenance_cost: float
    business_value: float
    risk_assessment: str
    retention_recommendation: str
    deletion_justification: Optional[str] = None
    retention_justification: Optional[str] = None

@dataclass
class StructureComparison:
    """结构对比结果"""
    timestamp: str
    files_added: List[FileAnalysis]
    files_removed: List[FileAnalysis]
    files_modified: List[Tuple[FileAnalysis, FileAnalysis]]
    directories_added: List[str]
    directories_removed: List[str]
    structure_changes: Dict[str, Any]
    impact_analysis: Dict[str, Any]
    recommendations: List[str]

class ProjectStructureAnalyzer:
    """项目结构分析器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.cache_dir = self.project_root / ".iflow" / "cache" / "structure_analysis"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    async def analyze_and_compare(self, baseline_snapshot: Optional[str] = None) -> StructureComparison:
        """分析并对比项目结构"""
        print("🔍 开始项目结构对比分析...")
        
        # 1. 获取当前项目结构
        current_structure = await self._analyze_current_structure()
        
        # 2. 加载基线结构（如果存在）
        baseline_structure = await self._load_baseline_structure(baseline_snapshot)
        
        # 3. 执行对比分析
        comparison = await self._compare_structures(current_structure, baseline_structure)
        
        # 4. 生成影响分析
        impact_analysis = await self._analyze_impact(comparison)
        comparison.impact_analysis = impact_analysis
        
        # 5. 生成推荐建议
        recommendations = await self._generate_recommendations(comparison)
        comparison.recommendations = recommendations
        
        # 6. 保存当前快照作为新的基线
        await self._save_structure_snapshot(current_structure)
        
        print("✅ 项目结构对比分析完成")
        return comparison
    
    async def _analyze_current_structure(self) -> Dict[str, Any]:
        """分析当前项目结构"""
        print("📊 分析当前项目结构...")
        
        structure = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "files": {},
            "directories": set(),
            "statistics": {},
            "dependencies": {},
            "complexity_metrics": {}
        }
        
        total_files = 0
        total_size = 0
        python_files = 0
        test_files = 0
        
        # 遍历项目文件
        for root, dirs, files in os.walk(self.project_root):
            # 跳过特定目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            rel_root = os.path.relpath(root, self.project_root)
            if rel_root == '.':
                rel_root = 'root'
            
            structure["directories"].add(rel_root)
            
            for file in files:
                if not file.startswith('.') and not file.endswith('.pyc'):
                    file_path = Path(root) / file
                    rel_path = str(file_path.relative_to(self.project_root))
                    
                    try:
                        file_analysis = await self._analyze_file(file_path)
                        structure["files"][rel_path] = file_analysis
                        
                        total_files += 1
                        total_size += file_analysis.size
                        
                        if file.endswith('.py'):
                            python_files += 1
                        if 'test' in file.lower():
                            test_files += 1
                        
                        # 分析依赖关系
                        deps = await self._analyze_file_dependencies(file_path)
                        structure["dependencies"][rel_path] = deps
                        
                        # 计算复杂度指标
                        complexity = await self._calculate_file_complexity(file_path)
                        structure["complexity_metrics"][rel_path] = complexity
                        
                    except Exception as e:
                        print(f"⚠️ 分析文件失败 {rel_path}: {e}")
        
        structure["statistics"] = {
            "total_files": total_files,
            "total_size": total_size,
            "python_files": python_files,
            "test_files": test_files,
            "directories": len(structure["directories"])
        }
        
        return structure
    
    async def _analyze_file(self, file_path: Path) -> FileAnalysis:
        """分析单个文件"""
        try:
            with open(file_path, 'rb') as f:
                content_bytes = f.read()
            
            # 计算文件哈希
            file_hash = hashlib.md5(content_bytes).hexdigest()
            
            # 如果是Python文件，进行深度分析
            if file_path.suffix == '.py':
                try:
                    content = content_bytes.decode('utf-8')
                    tree = ast.parse(content)
                    
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
                    
                    # 计算各项指标
                    complexity_score = await self._calculate_complexity_score(content)
                    functionality_score = await self._calculate_functionality_score(content, file_path.name)
                    maintenance_cost = await self._calculate_maintenance_cost(content, functions, classes)
                    business_value = await self._calculate_business_value(content, file_path.name)
                    risk_assessment = await self._assess_file_risk(content, file_path.name)
                    retention_recommendation = await self._recommend_retention(business_value, maintenance_cost, risk_assessment)
                    
                    return FileAnalysis(
                        path=str(file_path.relative_to(self.project_root)),
                        name=file_path.name,
                        size=len(content_bytes),
                        modified_time=file_path.stat().st_mtime,
                        file_hash=file_hash,
                        file_type='python',
                        functions=functions,
                        classes=classes,
                        imports=imports,
                        complexity_score=complexity_score,
                        dependencies=[],
                        functionality_score=functionality_score,
                        maintenance_cost=maintenance_cost,
                        business_value=business_value,
                        risk_assessment=risk_assessment,
                        retention_recommendation=retention_recommendation
                    )
                    
                except Exception as e:
                    print(f"⚠️ Python文件分析失败 {file_path}: {e}")
            
            # 非Python文件的基本分析
            return FileAnalysis(
                path=str(file_path.relative_to(self.project_root)),
                name=file_path.name,
                size=len(content_bytes),
                modified_time=file_path.stat().st_mtime,
                file_hash=file_hash,
                file_type=file_path.suffix[1:] if file_path.suffix else 'unknown',
                functions=[],
                classes=[],
                imports=[],
                complexity_score=0.0,
                dependencies=[],
                functionality_score=0.5,
                maintenance_cost=0.5,
                business_value=0.5,
                risk_assessment="low",
                retention_recommendation="keep"
            )
            
        except Exception as e:
            print(f"⚠️ 文件分析失败 {file_path}: {e}")
            return FileAnalysis(
                path=str(file_path.relative_to(self.project_root)),
                name=file_path.name,
                size=0,
                modified_time=0,
                file_hash="",
                file_type='error',
                functions=[],
                classes=[],
                imports=[],
                complexity_score=0.0,
                dependencies=[],
                functionality_score=0.0,
                maintenance_cost=0.0,
                business_value=0.0,
                risk_assessment="error",
                retention_recommendation="review"
            )
    
    async def _calculate_complexity_score(self, content: str) -> float:
        """计算复杂度分数"""
        lines = content.split('\n')
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        tree = ast.parse(content)
        
        # 圈复杂度
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return float(complexity + code_lines * 0.1)
    
    async def _calculate_functionality_score(self, content: str, filename: str) -> float:
        """计算功能价值分数"""
        score = 0.5  # 基础分数
        
        # 基于文件名
        if 'engine' in filename.lower():
            score += 0.3
        if 'core' in filename.lower():
            score += 0.3
        if 'main' in filename.lower():
            score += 0.2
        if 'test' in filename.lower():
            score += 0.1
        if 'util' in filename.lower():
            score += 0.1
        if 'cache' in filename.lower():
            score += 0.15
        if 'security' in filename.lower():
            score += 0.25
        
        # 基于内容
        if 'class' in content:
            score += 0.1
        if 'def ' in content:
            score += 0.1
        if 'async def' in content:
            score += 0.15
        if 'import' in content:
            score += 0.05
        
        return min(score, 1.0)
    
    async def _calculate_maintenance_cost(self, content: str, functions: List[str], classes: List[str]) -> float:
        """计算维护成本"""
        cost = 0.1  # 基础成本
        
        # 基于代码量
        lines = len(content.split('\n'))
        cost += lines * 0.001
        
        # 基于复杂度
        cost += len(functions) * 0.02
        cost += len(classes) * 0.03
        
        # 基于导入数量
        imports = content.count('import')
        cost += imports * 0.01
        
        # 基于注释质量
        comment_lines = content.count('#')
        if comment_lines > 0:
            cost -= comment_lines * 0.0005
        
        return min(cost, 1.0)
    
    async def _calculate_business_value(self, content: str, filename: str) -> float:
        """计算业务价值"""
        value = 0.3  # 基础价值
        
        # 核心模块价值更高
        if any(keyword in filename.lower() for keyword in ['engine', 'core', 'main', 'kernel']):
            value += 0.4
        
        # 安全功能价值高
        if 'security' in filename.lower():
            value += 0.3
        
        # 性能相关价值高
        if any(keyword in filename.lower() for keyword in ['cache', 'optimize', 'performance']):
            value += 0.2
        
        # 用户接口价值高
        if any(keyword in filename.lower() for keyword in ['api', 'interface', 'ui', 'cli']):
            value += 0.25
        
        # 数据处理价值中等
        if any(keyword in filename.lower() for keyword in ['data', 'process', 'transform']):
            value += 0.15
        
        # 测试价值较低但重要
        if 'test' in filename.lower():
            value += 0.1
        
        return min(value, 1.0)
    
    async def _assess_file_risk(self, content: str, filename: str) -> str:
        """评估文件风险"""
        risk_score = 0
        
        # 复杂度风险
        if len(content) > 1000:
            risk_score += 1
        if content.count('class ') > 5:
            risk_score += 1
        if content.count('def ') > 20:
            risk_score += 1
        
        # 安全风险
        if any(keyword in content for keyword in ['eval', 'exec', 'compile']):
            risk_score += 3
        if 'password' in content.lower() or 'secret' in content.lower():
            risk_score += 2
        
        # 依赖风险
        if content.count('import') > 10:
            risk_score += 1
        
        # 确定风险等级
        if risk_score >= 4:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"
    
    async def _recommend_retention(self, business_value: float, maintenance_cost: float, risk_assessment: str) -> str:
        """推荐保留或删除"""
        # 计算净价值
        net_value = business_value - maintenance_cost
        
        # 风险调整
        if risk_assessment == "high":
            net_value -= 0.2
        elif risk_assessment == "medium":
            net_value -= 0.1
        
        # 决策
        if net_value >= 0.3:
            return "keep"
        elif net_value >= 0.1:
            return "review"
        else:
            return "consider_remove"
    
    async def _analyze_file_dependencies(self, file_path: Path) -> List[str]:
        """分析文件依赖"""
        if file_path.suffix != '.py':
            return []
        
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
    
    async def _calculate_file_complexity(self, file_path: Path) -> Dict[str, float]:
        """计算文件复杂度指标"""
        if file_path.suffix != '.py':
            return {"complexity": 0.0}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            metrics = {
                "cyclomatic_complexity": 1,
                "cognitive_complexity": 0,
                "halstead_volume": 0.0,
                "maintainability_index": 100.0
            }
            
            # 圈复杂度
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.With)):
                    metrics["cyclomatic_complexity"] += 1
                elif isinstance(node, ast.ExceptHandler):
                    metrics["cyclomatic_complexity"] += 1
                elif isinstance(node, ast.BoolOp):
                    metrics["cyclomatic_complexity"] += len(node.values) - 1
            
            # 简化的可维护性指数
            lines = len(content.split('\n'))
            metrics["maintainability_index"] = max(0, 100 - metrics["cyclomatic_complexity"] * 2 - lines * 0.1)
            
            return metrics
            
        except Exception:
            return {"complexity": 0.0}
    
    async def _load_baseline_structure(self, baseline_snapshot: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """加载基线结构"""
        if baseline_snapshot:
            snapshot_file = self.cache_dir / f"structure_snapshot_{baseline_snapshot}.json"
        else:
            # 加载最新的快照
            snapshot_files = list(self.cache_dir.glob("structure_snapshot_*.json"))
            if not snapshot_files:
                return None
            
            snapshot_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            snapshot_file = snapshot_files[0]
        
        try:
            with open(snapshot_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 加载基线结构失败: {e}")
            return None
    
    async def _compare_structures(self, current: Dict[str, Any], baseline: Optional[Dict[str, Any]]) -> StructureComparison:
        """对比结构"""
        print("🔄 执行结构对比...")
        
        comparison = StructureComparison(
            timestamp=datetime.now().isoformat(),
            files_added=[],
            files_removed=[],
            files_modified=[],
            directories_added=[],
            directories_removed=[],
            structure_changes={},
            impact_analysis={},
            recommendations=[]
        )
        
        if not baseline:
            print("ℹ️ 未找到基线结构，创建初始快照")
            return comparison
        
        current_files = set(current["files"].keys())
        baseline_files = set(baseline["files"].keys())
        
        # 找出新增文件
        added_paths = current_files - baseline_files
        for path in added_paths:
            comparison.files_added.append(current["files"][path])
        
        # 找出删除文件
        removed_paths = baseline_files - current_files
        for path in removed_paths:
            comparison.files_removed.append(baseline["files"][path])
        
        # 找出修改文件
        common_paths = current_files & baseline_files
        for path in common_paths:
            current_file = current["files"][path]
            baseline_file = baseline["files"][path]
            
            if current_file["file_hash"] != baseline_file["file_hash"]:
                comparison.files_modified.append((current_file, baseline_file))
        
        # 目录变化
        current_dirs = set(current["directories"])
        baseline_dirs = set(baseline["directories"])
        
        comparison.directories_added = list(current_dirs - baseline_dirs)
        comparison.directories_removed = list(baseline_dirs - current_dirs)
        
        # 结构变化统计
        comparison.structure_changes = {
            "files_added_count": len(comparison.files_added),
            "files_removed_count": len(comparison.files_removed),
            "files_modified_count": len(comparison.files_modified),
            "directories_added_count": len(comparison.directories_added),
            "directories_removed_count": len(comparison.directories_removed),
            "total_files_before": len(baseline_files),
            "total_files_after": len(current_files)
        }
        
        return comparison
    
    async def _analyze_impact(self, comparison: StructureComparison) -> Dict[str, Any]:
        """分析影响"""
        print("📈 分析变化影响...")
        
        impact = {
            "functional_impact": {},
            "performance_impact": {},
            "security_impact": {},
            "maintenance_impact": {},
            "dependency_impact": {},
            "overall_risk": "low"
        }
        
        # 功能影响分析
        functional_score = 0
        for file in comparison.files_removed:
            functional_score += file.get("business_value", 0)
        
        for current_file, baseline_file in comparison.files_modified:
            # 简化的影响计算
            functional_score += abs(current_file.get("functionality_score", 0) - baseline_file.get("functionality_score", 0))
        
        impact["functional_impact"] = {
            "score": functional_score,
            "level": "high" if functional_score > 0.5 else "medium" if functional_score > 0.2 else "low"
        }
        
        # 性能影响分析
        performance_impact = 0
        for file in comparison.files_added:
            if "engine" in file["name"].lower() or "cache" in file["name"].lower():
                performance_impact += 0.3
        
        impact["performance_impact"] = {
            "score": performance_impact,
            "level": "high" if performance_impact > 0.3 else "medium" if performance_impact > 0.1 else "low"
        }
        
        # 安全影响分析
        security_impact = 0
        for file in comparison.files_removed:
            if file.get("risk_assessment") == "low":
                security_impact += 0.1
            elif file.get("risk_assessment") == "medium":
                security_impact += 0.2
            elif file.get("risk_assessment") == "high":
                security_impact += 0.3
        
        impact["security_impact"] = {
            "score": security_impact,
            "level": "high" if security_impact > 0.3 else "medium" if security_impact > 0.1 else "low"
        }
        
        # 维护影响分析
        maintenance_impact = 0
        for file in comparison.files_added:
            maintenance_impact += file.get("maintenance_cost", 0)
        
        for current_file, baseline_file in comparison.files_modified:
            maintenance_impact += abs(current_file.get("maintenance_cost", 0) - baseline_file.get("maintenance_cost", 0))
        
        impact["maintenance_impact"] = {
            "score": maintenance_impact,
            "level": "high" if maintenance_impact > 0.5 else "medium" if maintenance_impact > 0.2 else "low"
        }
        
        # 依赖影响分析
        dependency_changes = len(comparison.files_added) + len(comparison.files_removed)
        impact["dependency_impact"] = {
            "score": dependency_changes * 0.1,
            "level": "high" if dependency_changes > 10 else "medium" if dependency_changes > 5 else "low"
        }
        
        # 整体风险评估
        risk_scores = [
            impact["functional_impact"]["score"],
            impact["performance_impact"]["score"],
            impact["security_impact"]["score"],
            impact["maintenance_impact"]["score"],
            impact["dependency_impact"]["score"]
        ]
        
        total_risk = sum(risk_scores)
        if total_risk > 1.5:
            impact["overall_risk"] = "high"
        elif total_risk > 0.8:
            impact["overall_risk"] = "medium"
        else:
            impact["overall_risk"] = "low"
        
        return impact
    
    async def _generate_recommendations(self, comparison: StructureComparison) -> List[str]:
        """生成推荐建议"""
        recommendations = []
        
        # 基于文件变化的建议
        if comparison.files_removed:
            high_value_removed = [f for f in comparison.files_removed if f.get("business_value", 0) > 0.5]
            if high_value_removed:
                recommendations.append(f"警告：删除了{len(high_value_removed)}个高价值文件，建议重新评估")
        
        if comparison.files_added:
            high_cost_added = [f for f in comparison.files_added if f.get("maintenance_cost", 0) > 0.7]
            if high_cost_added:
                recommendations.append(f"注意：新增了{len(high_cost_added)}个高维护成本文件，需要关注")
        
        # 基于影响的建议
        impact = comparison.impact_analysis
        
        if impact["functional_impact"]["level"] == "high":
            recommendations.append("功能影响较大，建议进行回归测试")
        
        if impact["security_impact"]["level"] == "high":
            recommendations.append("安全影响较大，建议进行安全审计")
        
        if impact["performance_impact"]["level"] == "high":
            recommendations.append("性能影响较大，建议进行性能测试")
        
        if impact["maintenance_impact"]["level"] == "high":
            recommendations.append("维护成本增加较多，建议优化代码结构")
        
        # 基于整体风险的建议
        if impact["overall_risk"] == "high":
            recommendations.append("整体风险较高，建议分阶段部署")
        elif impact["overall_risk"] == "medium":
            recommendations.append("整体风险中等，建议加强监控")
        
        # 结构优化建议
        if comparison.structure_changes["files_added_count"] > comparison.structure_changes["files_removed_count"] * 2:
            recommendations.append("文件增长较快，建议检查是否有冗余代码")
        
        return recommendations
    
    async def _save_structure_snapshot(self, structure: Dict[str, Any]):
        """保存结构快照"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_file = self.cache_dir / f"structure_snapshot_{timestamp}.json"
        
        try:
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(structure, f, ensure_ascii=False, indent=2)
            
            print(f"📸 结构快照已保存: {snapshot_file}")
            
            # 清理旧快照（保留最近10个）
            snapshot_files = list(self.cache_dir.glob("structure_snapshot_*.json"))
            snapshot_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for old_snapshot in snapshot_files[10:]:
                old_snapshot.unlink()
                
        except Exception as e:
            print(f"⚠️ 保存结构快照失败: {e}")
    
    async def generate_detailed_report(self, comparison: StructureComparison) -> str:
        """生成详细报告"""
        report = []
        report.append("# 项目结构对比分析报告")
        report.append(f"生成时间: {comparison.timestamp}")
        report.append("")
        
        # 变化概览
        report.append("## 📊 变化概览")
        changes = comparison.structure_changes
        report.append(f"- 新增文件: {changes['files_added_count']}个")
        report.append(f"- 删除文件: {changes['files_removed_count']}个")
        report.append(f"- 修改文件: {changes['files_modified_count']}个")
        report.append(f"- 新增目录: {changes['directories_added_count']}个")
        report.append(f"- 删除目录: {changes['directories_removed_count']}个")
        report.append("")
        
        # 新增文件详情
        if comparison.files_added:
            report.append("## 📁 新增文件")
            for file in comparison.files_added:
                report.append(f"### {file['name']}")
                report.append(f"- 路径: {file['path']}")
                report.append(f"- 大小: {file['size']}字节")
                report.append(f"- 功能价值: {file.get('functionality_score', 0):.2f}")
                report.append(f"- 维护成本: {file.get('maintenance_cost', 0):.2f}")
                report.append(f"- 推荐操作: {file.get('retention_recommendation', 'unknown')}")
                report.append("")
        
        # 删除文件详情
        if comparison.files_removed:
            report.append("## 🗑️ 删除文件")
            for file in comparison.files_removed:
                report.append(f"### {file['name']}")
                report.append(f"- 路径: {file['path']}")
                report.append(f"- 大小: {file['size']}字节")
                report.append(f"- 业务价值: {file.get('business_value', 0):.2f}")
                report.append(f"- 风险评估: {file.get('risk_assessment', 'unknown')}")
                report.append(f"- 删除理由: {file.get('deletion_justification', '未提供')}")
                report.append("")
        
        # 影响分析
        report.append("## 📈 影响分析")
        impact = comparison.impact_analysis
        report.append(f"### 功能影响: {impact['functional_impact']['level']} ({impact['functional_impact']['score']:.2f})")
        report.append(f"### 性能影响: {impact['performance_impact']['level']} ({impact['performance_impact']['score']:.2f})")
        report.append(f"### 安全影响: {impact['security_impact']['level']} ({impact['security_impact']['score']:.2f})")
        report.append(f"### 维护影响: {impact['maintenance_impact']['level']} ({impact['maintenance_impact']['score']:.2f})")
        report.append(f"### 依赖影响: {impact['dependency_impact']['level']} ({impact['dependency_impact']['score']:.2f})")
        report.append(f"### 整体风险: {impact['overall_risk']}")
        report.append("")
        
        # 推荐建议
        if comparison.recommendations:
            report.append("## 💡 推荐建议")
            for i, rec in enumerate(comparison.recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        return "\n".join(report)

# 使用示例
async def main():
    """主函数"""
    project_root = "."
    
    analyzer = ProjectStructureAnalyzer(project_root)
    comparison = await analyzer.analyze_and_compare()
    
    # 生成详细报告
    report = await analyzer.generate_detailed_report(comparison)
    
    # 保存报告
    report_file = Path(project_root) / "structure_comparison_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📋 详细报告已保存到: {report_file}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())