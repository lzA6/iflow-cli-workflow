#!/usr/bin/env python3
"""
功能特点分析模块
深入分析每个文件的功能特点、优缺点、价值评估和替代方案
"""

import os
import re
import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
from enum import Enum

class FeatureCategory(Enum):
    """功能类别"""
    CORE_ENGINE = "core_engine"
    WORKFLOW_SYSTEM = "workflow_system"
    KNOWLEDGE_BASE = "knowledge_base"
    UTILITY_MODULE = "utility_module"
    TEST_MODULE = "test_module"
    CONFIG_MODULE = "config_module"
    API_MODULE = "api_module"
    SECURITY_MODULE = "security_module"
    PERFORMANCE_MODULE = "performance_module"
    UI_MODULE = "ui_module"

class ValueLevel(Enum):
    """价值等级"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"

@dataclass
class FeatureCharacteristic:
    """功能特征"""
    name: str
    description: str
    category: FeatureCategory
    value_level: ValueLevel
    uniqueness: float  # 独特性 0-1
    complexity: float  # 复杂度 0-1
    maturity: float    # 成熟度 0-1
    usage_frequency: str  # 使用频率
    user_impact: str     # 用户影响
    business_value: str  # 业务价值
    technical_debt: float  # 技术债务 0-1

@dataclass
class Advantage:
    """优势"""
    category: str
    description: str
    impact_level: str
    evidence: List[str]
    quantification: Optional[str]

@dataclass
class Disadvantage:
    """劣势"""
    category: str
    description: str
    impact_level: str
    evidence: List[str]
    mitigation: Optional[str]

@dataclass
class Alternative:
    """替代方案"""
    name: str
    description: str
    feasibility: float  # 可行性 0-1
    cost_estimate: str
    pros: List[str]
    cons: List[str]
    implementation_effort: str

@dataclass
class FunctionalityAnalysis:
    """功能分析结果"""
    file_path: str
    feature_characteristics: List[FeatureCharacteristic]
    advantages: List[Advantage]
    disadvantages: List[Disadvantage]
    alternatives: List[Alternative]
    retention_justification: str
    removal_justification: Optional[str]
    replacement_options: List[str]
    integration_points: List[str]
    dependencies: List[str]
    dependents: List[str]
    overall_assessment: str
    recommendation: str

class FeatureAnalysisModule:
    """功能特点分析模块"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.analysis_cache = {}
        self.feature_patterns = self._load_feature_patterns()
        self.value_assessment_criteria = self._load_value_criteria()
        
    async def analyze_comprehensive_features(self, file_path: str) -> FunctionalityAnalysis:
        """综合功能分析"""
        print(f"🔍 开始综合功能分析: {file_path}")
        
        # 1. 基础文件分析
        print("📁 分析基础文件信息...")
        basic_info = await self._analyze_basic_file_info(file_path)
        
        # 2. 功能特征识别
        print("🎯 识别功能特征...")
        feature_characteristics = await self._identify_feature_characteristics(file_path, basic_info)
        
        # 3. 优势分析
        print("💪 分析优势...")
        advantages = await self._analyze_advantages(file_path, feature_characteristics)
        
        # 4. 劣势分析
        print("⚠️ 分析劣势...")
        disadvantages = await self._analyze_disadvantages(file_path, feature_characteristics)
        
        # 5. 替代方案分析
        print("🔄 分析替代方案...")
        alternatives = await self._analyze_alternatives(file_path, feature_characteristics)
        
        # 6. 依赖关系分析
        print("🔗 分析依赖关系...")
        dependencies, dependents = await self._analyze_dependencies(file_path)
        
        # 7. 集成点分析
        print("🔌 分析集成点...")
        integration_points = await self._analyze_integration_points(file_path)
        
        # 8. 保留/删除理由生成
        print("⚖️ 生成决策理由...")
        retention_justification = await self._generate_retention_justification(
            file_path, feature_characteristics, advantages, disadvantages
        )
        removal_justification = await self._generate_removal_justification(
            file_path, feature_characteristics, disadvantages
        )
        
        # 9. 替换选项分析
        print("🔄 分析替换选项...")
        replacement_options = await self._analyze_replacement_options(file_path, alternatives)
        
        # 10. 整体评估
        print("📊 进行整体评估...")
        overall_assessment = await self._perform_overall_assessment(
            feature_characteristics, advantages, disadvantages
        )
        
        # 11. 推荐建议
        print("💡 生成推荐建议...")
        recommendation = await self._generate_recommendation(
            file_path, overall_assessment, alternatives
        )
        
        # 12. 构建分析结果
        analysis_result = FunctionalityAnalysis(
            file_path=file_path,
            feature_characteristics=feature_characteristics,
            advantages=advantages,
            disadvantages=disadvantages,
            alternatives=alternatives,
            retention_justification=retention_justification,
            removal_justification=removal_justification,
            replacement_options=replacement_options,
            integration_points=integration_points,
            dependencies=dependencies,
            dependents=dependents,
            overall_assessment=overall_assessment,
            recommendation=recommendation
        )
        
        print(f"✅ 功能分析完成: {file_path}")
        return analysis_result
    
    async def _analyze_basic_file_info(self, file_path: str) -> Dict[str, Any]:
        """分析基础文件信息"""
        full_path = self.project_root / file_path
        
        if not full_path.exists():
            return {"error": "文件不存在"}
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 基本统计
            lines = content.split('\n')
            code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            
            # AST分析
            try:
                tree = ast.parse(content)
                functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend([alias.name for alias in node.names])
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        imports.extend([f"{module}.{alias.name}" for alias in node.names])
            except:
                functions = []
                classes = []
                imports = []
            
            return {
                "file_size": full_path.stat().st_size,
                "total_lines": len(lines),
                "code_lines": code_lines,
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "file_name": full_path.name,
                "file_extension": full_path.suffix,
                "last_modified": full_path.stat().st_mtime
            }
            
        except Exception as e:
            return {"error": f"文件读取失败: {e}"}
    
    async def _identify_feature_characteristics(self, file_path: str, basic_info: Dict[str, Any]) -> List[FeatureCharacteristic]:
        """识别功能特征"""
        characteristics = []
        
        if "error" in basic_info:
            return characteristics
        
        file_name = basic_info["file_name"].lower()
        content = await self._read_file_content(file_path)
        
        # 基于文件名和内容识别特征
        for pattern_name, pattern_config in self.feature_patterns.items():
            if await self._matches_pattern(file_name, content, pattern_config):
                characteristic = await self._create_characteristic_from_pattern(
                    pattern_name, pattern_config, basic_info
                )
                characteristics.append(characteristic)
        
        # 基于代码结构识别特征
        structure_characteristics = await self._analyze_structure_characteristics(basic_info, content)
        characteristics.extend(structure_characteristics)
        
        return characteristics
    
    async def _matches_pattern(self, file_name: str, content: str, pattern_config: Dict[str, Any]) -> bool:
        """检查是否匹配模式"""
        # 检查文件名模式
        if "filename_patterns" in pattern_config:
            for pattern in pattern_config["filename_patterns"]:
                if re.search(pattern, file_name):
                    return True
        
        # 检查内容模式
        if "content_patterns" in pattern_config:
            for pattern in pattern_config["content_patterns"]:
                if re.search(pattern, content, re.IGNORECASE):
                    return True
        
        # 检查关键词
        if "keywords" in pattern_config:
            for keyword in pattern_config["keywords"]:
                if keyword.lower() in content.lower():
                    return True
        
        return False
    
    async def _create_characteristic_from_pattern(self, pattern_name: str, 
                                                pattern_config: Dict[str, Any], 
                                                basic_info: Dict[str, Any]) -> FeatureCharacteristic:
        """从模式创建特征"""
        category = FeatureCategory(pattern_config.get("category", "utility_module"))
        value_level = ValueLevel(pattern_config.get("value_level", "medium"))
        
        # 计算独特性
        uniqueness = await self._calculate_uniqueness(pattern_name, basic_info)
        
        # 计算复杂度
        complexity = await self._calculate_complexity(basic_info)
        
        # 评估成熟度
        maturity = await self._assess_maturity(pattern_name, basic_info)
        
        return FeatureCharacteristic(
            name=pattern_config["name"],
            description=pattern_config["description"],
            category=category,
            value_level=value_level,
            uniqueness=uniqueness,
            complexity=complexity,
            maturity=maturity,
            usage_frequency=pattern_config.get("usage_frequency", "unknown"),
            user_impact=pattern_config.get("user_impact", "medium"),
            business_value=pattern_config.get("business_value", "medium"),
            technical_debt=pattern_config.get("technical_debt", 0.3)
        )
    
    async def _analyze_structure_characteristics(self, basic_info: Dict[str, Any], content: str) -> List[FeatureCharacteristic]:
        """分析结构特征"""
        characteristics = []
        
        # 基于函数数量
        function_count = len(basic_info.get("functions", []))
        if function_count > 10:
            characteristics.append(FeatureCharacteristic(
                name="多功能模块",
                description=f"包含{function_count}个函数的复杂模块",
                category=FeatureCategory.UTILITY_MODULE,
                value_level=ValueLevel.MEDIUM,
                uniqueness=0.6,
                complexity=0.8,
                maturity=0.7,
                usage_frequency="medium",
                user_impact="medium",
                business_value="medium",
                technical_debt=0.4
            ))
        
        # 基于类数量
        class_count = len(basic_info.get("classes", []))
        if class_count > 0:
            characteristics.append(FeatureCharacteristic(
                name="面向对象设计",
                description=f"包含{class_count}个类的面向对象模块",
                category=FeatureCategory.CORE_ENGINE,
                value_level=ValueLevel.HIGH,
                uniqueness=0.7,
                complexity=0.6,
                maturity=0.8,
                usage_frequency="high",
                user_impact="high",
                business_value="high",
                technical_debt=0.2
            ))
        
        # 基于异步特性
        if "async def" in content:
            characteristics.append(FeatureCharacteristic(
                name="异步处理能力",
                description="支持异步编程的模块",
                category=FeatureCategory.PERFORMANCE_MODULE,
                value_level=ValueLevel.HIGH,
                uniqueness=0.8,
                complexity=0.7,
                maturity=0.8,
                usage_frequency="high",
                user_impact="high",
                business_value="high",
                technical_debt=0.3
            ))
        
        return characteristics
    
    async def _calculate_uniqueness(self, pattern_name: str, basic_info: Dict[str, Any]) -> float:
        """计算独特性"""
        # 简化的独特性计算
        file_name = basic_info["file_name"].lower()
        
        # 基于文件名的独特性
        unique_indicators = ["engine", "kernel", "core", "quantum", "evolution"]
        uniqueness_score = 0.5  # 基础分数
        
        for indicator in unique_indicators:
            if indicator in file_name:
                uniqueness_score += 0.1
        
        # 基于函数名的独特性
        functions = basic_info.get("functions", [])
        unique_functions = [f for f in functions if any(keyword in f.lower() for keyword in ["quantum", "evolution", "intelligent", "smart"])]
        if unique_functions:
            uniqueness_score += 0.2
        
        return min(1.0, uniqueness_score)
    
    async def _calculate_complexity(self, basic_info: Dict[str, Any]) -> float:
        """计算复杂度"""
        code_lines = basic_info.get("code_lines", 0)
        function_count = len(basic_info.get("functions", []))
        class_count = len(basic_info.get("classes", []))
        import_count = len(basic_info.get("imports", []))
        
        # 简化的复杂度计算
        complexity_score = 0.0
        
        # 基于代码行数
        if code_lines > 500:
            complexity_score += 0.3
        elif code_lines > 200:
            complexity_score += 0.2
        elif code_lines > 100:
            complexity_score += 0.1
        
        # 基于函数数量
        if function_count > 20:
            complexity_score += 0.3
        elif function_count > 10:
            complexity_score += 0.2
        elif function_count > 5:
            complexity_score += 0.1
        
        # 基于类数量
        if class_count > 5:
            complexity_score += 0.2
        elif class_count > 2:
            complexity_score += 0.1
        
        # 基于导入数量
        if import_count > 10:
            complexity_score += 0.2
        elif import_count > 5:
            complexity_score += 0.1
        
        return min(1.0, complexity_score)
    
    async def _assess_maturity(self, pattern_name: str, basic_info: Dict[str, Any]) -> float:
        """评估成熟度"""
        # 简化的成熟度评估
        maturity_score = 0.5  # 基础分数
        
        file_name = basic_info["file_name"].lower()
        
        # 基于版本号
        if re.search(r'v\d+_\d+', file_name):
            maturity_score += 0.2
        
        # 基于文档注释
        try:
            content = await self._read_file_content(basic_info["file_path"])
            docstring_count = content.count('"""') + content.count("'''")
            if docstring_count > 0:
                maturity_score += 0.1
        except:
            pass
        
        # 基于错误处理
        try:
            content = await self._read_file_content(basic_info["file_path"])
            if "try:" in content and "except" in content:
                maturity_score += 0.2
        except:
            pass
        
        return min(1.0, maturity_score)
    
    async def _analyze_advantages(self, file_path: str, 
                                feature_characteristics: List[FeatureCharacteristic]) -> List[Advantage]:
        """分析优势"""
        advantages = []
        
        # 基于特征分析优势
        for characteristic in feature_characteristics:
            category_advantages = await self._generate_advantages_from_characteristic(characteristic)
            advantages.extend(category_advantages)
        
        # 基于代码质量分析优势
        quality_advantages = await self._analyze_quality_advantages(file_path)
        advantages.extend(quality_advantages)
        
        # 基于架构分析优势
        architecture_advantages = await self._analyze_architecture_advantages(file_path)
        advantages.extend(architecture_advantages)
        
        return advantages
    
    async def _generate_advantages_from_characteristic(self, characteristic: FeatureCharacteristic) -> List[Advantage]:
        """从特征生成优势"""
        advantages = []
        
        if characteristic.category == FeatureCategory.CORE_ENGINE:
            advantages.append(Advantage(
                category="核心功能",
                description=f"提供{characteristic.name}的核心功能",
                impact_level="high",
                evidence=[f"特征类别: {characteristic.category.value}"],
                quantification=f"价值等级: {characteristic.value_level.value}"
            ))
        
        if characteristic.uniqueness > 0.7:
            advantages.append(Advantage(
                category="独特性",
                description=f"具有{characteristic.uniqueness:.1%}的独特性",
                impact_level="high",
                evidence=["独特性评分高"],
                quantification=f"独特性: {characteristic.uniqueness:.1%}"
            ))
        
        if characteristic.maturity > 0.7:
            advantages.append(Advantage(
                category="成熟度",
                description=f"代码成熟度高({characteristic.maturity:.1%})",
                impact_level="medium",
                evidence=["成熟度评分高"],
                quantification=f"成熟度: {characteristic.maturity:.1%}"
            ))
        
        return advantages
    
    async def _analyze_quality_advantages(self, file_path: str) -> List[Advantage]:
        """分析质量优势"""
        advantages = []
        
        try:
            content = await self._read_file_content(file_path)
            
            # 检查文档完整性
            docstring_count = content.count('"""') + content.count("'''")
            if docstring_count > 0:
                advantages.append(Advantage(
                    category="文档完整性",
                    description=f"包含{docstring_count}个文档字符串",
                    impact_level="medium",
                    evidence=["发现文档字符串"],
                    quantification=f"文档字符串数量: {docstring_count}"
                ))
            
            # 检查错误处理
            if "try:" in content and "except" in content:
                advantages.append(Advantage(
                    category="错误处理",
                    description="包含异常处理机制",
                    impact_level="high",
                    evidence=["发现try-except块"],
                    quantification="具备错误处理能力"
                ))
            
            # 检查模块化设计
            if "def " in content:
                function_count = content.count("def ")
                if function_count > 1:
                    advantages.append(Advantage(
                        category="模块化设计",
                        description=f"包含{function_count}个函数，模块化程度高",
                        impact_level="medium",
                        evidence=[f"函数数量: {function_count}"],
                        quantification=f"模块化程度: {function_count}个函数"
                    ))
        
        except Exception as e:
            advantages.append(Advantage(
                category="分析限制",
                description=f"质量分析受限: {e}",
                impact_level="low",
                evidence=["分析错误"],
                quantification=None
            ))
        
        return advantages
    
    async def _analyze_architecture_advantages(self, file_path: str) -> List[Advantage]:
        """分析架构优势"""
        advantages = []
        
        try:
            content = await self._read_file_content(file_path)
            
            # 检查面向对象设计
            if "class " in content:
                class_count = content.count("class ")
                advantages.append(Advantage(
                    category="面向对象设计",
                    description=f"采用面向对象设计，包含{class_count}个类",
                    impact_level="high",
                    evidence=[f"类数量: {class_count}"],
                    quantification=f"面向对象程度: {class_count}个类"
                ))
            
            # 检查异步设计
            if "async def" in content:
                async_function_count = content.count("async def")
                advantages.append(Advantage(
                    category="异步设计",
                    description=f"支持异步编程，包含{async_function_count}个异步函数",
                    impact_level="high",
                    evidence=[f"异步函数数量: {async_function_count}"],
                    quantification=f"异步程度: {async_function_count}个异步函数"
                ))
            
            # 检查接口设计
            if "import" in content:
                import_count = content.count("import")
                if import_count > 0:
                    advantages.append(Advantage(
                        category="接口设计",
                        description=f"良好的模块接口设计，{import_count}个导入",
                        impact_level="medium",
                        evidence=[f"导入数量: {import_count}"],
                        quantification=f"接口复杂度: {import_count}个导入"
                    ))
        
        except Exception as e:
            advantages.append(Advantage(
                category="分析限制",
                description=f"架构分析受限: {e}",
                impact_level="low",
                evidence=["分析错误"],
                quantification=None
            ))
        
        return advantages
    
    async def _analyze_disadvantages(self, file_path: str, 
                                   feature_characteristics: List[FeatureCharacteristic]) -> List[Disadvantage]:
        """分析劣势"""
        disadvantages = []
        
        # 基于特征分析劣势
        for characteristic in feature_characteristics:
            category_disadvantages = await self._generate_disadvantages_from_characteristic(characteristic)
            disadvantages.extend(category_disadvantages)
        
        # 基于代码质量分析劣势
        quality_disadvantages = await self._analyze_quality_disadvantages(file_path)
        disadvantages.extend(quality_disadvantages)
        
        # 基于架构分析劣势
        architecture_disadvantages = await self._analyze_architecture_disadvantages(file_path)
        disadvantages.extend(architecture_disadvantages)
        
        return disadvantages
    
    async def _generate_disadvantages_from_characteristic(self, characteristic: FeatureCharacteristic) -> List[Disadvantage]:
        """从特征生成劣势"""
        disadvantages = []
        
        if characteristic.complexity > 0.7:
            disadvantages.append(Disadvantage(
                category="复杂度",
                description=f"复杂度较高({characteristic.complexity:.1%})，维护困难",
                impact_level="high",
                evidence=[f"复杂度评分: {characteristic.complexity:.1%}"],
                mitigation="重构简化，提高可维护性"
            ))
        
        if characteristic.technical_debt > 0.5:
            disadvantages.append(Disadvantage(
                category="技术债务",
                description=f"技术债务较高({characteristic.technical_debt:.1%})",
                impact_level="medium",
                evidence=[f"技术债务评分: {characteristic.technical_debt:.1%}"],
                mitigation="逐步重构，降低技术债务"
            ))
        
        if characteristic.maturity < 0.5:
            disadvantages.append(Disadvantage(
                category="成熟度",
                description=f"成熟度较低({characteristic.maturity:.1%})，可能存在不稳定因素",
                impact_level="medium",
                evidence=[f"成熟度评分: {characteristic.maturity:.1%}"],
                mitigation="加强测试，提升成熟度"
            ))
        
        return disadvantages
    
    async def _analyze_quality_disadvantages(self, file_path: str) -> List[Disadvantage]:
        """分析质量劣势"""
        disadvantages = []
        
        try:
            content = await self._read_file_content(file_path)
            lines = content.split('\n')
            
            # 检查代码长度
            if len(lines) > 500:
                disadvantages.append(Disadvantage(
                    category="代码长度",
                    description=f"代码过长({len(lines)}行)，难以维护",
                    impact_level="medium",
                    evidence=[f"代码行数: {len(lines)}"],
                    mitigation="拆分为多个模块"
                ))
            
            # 检查注释覆盖率
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            if code_lines > 0:
                comment_ratio = comment_lines / code_lines
                if comment_ratio < 0.1:
                    disadvantages.append(Disadvantage(
                        category="注释不足",
                        description=f"注释覆盖率低({comment_ratio:.1%})",
                        impact_level="medium",
                        evidence=[f"注释比例: {comment_ratio:.1%}"],
                        mitigation="增加代码注释"
                    ))
            
            # 检查硬编码
            if re.search(r'["\'][^"\']*["\']', content):
                hardcoded_strings = len(re.findall(r'["\'][^"\']*["\']', content))
                if hardcoded_strings > 10:
                    disadvantages.append(Disadvantage(
                        category="硬编码",
                        description=f"存在较多硬编码字符串({hardcoded_strings}个)",
                        impact_level="low",
                        evidence=[f"硬编码数量: {hardcoded_strings}"],
                        mitigation="使用配置文件或常量"
                    ))
        
        except Exception as e:
            disadvantages.append(Disadvantage(
                category="分析限制",
                description=f"质量分析受限: {e}",
                impact_level="low",
                evidence=["分析错误"],
                mitigation=None
            ))
        
        return disadvantages
    
    async def _analyze_architecture_disadvantages(self, file_path: str) -> List[Disadvantage]:
        """分析架构劣势"""
        disadvantages = []
        
        try:
            content = await self._read_file_content(file_path)
            
            # 检查依赖数量
            import_count = content.count("import")
            if import_count > 10:
                disadvantages.append(Disadvantage(
                    category="依赖过多",
                    description=f"外部依赖过多({import_count}个)，耦合度高",
                    impact_level="medium",
                    evidence=[f"导入数量: {import_count}"],
                    mitigation="减少不必要的依赖"
                ))
            
            # 检查函数长度
            functions = re.findall(r'def\s+\w+\s*\([^)]*\):', content)
            for func in functions:
                # 简化的函数长度检查
                func_start = content.find(func)
                if func_start != -1:
                    # 查找下一个函数或类定义
                    next_def = content.find('\ndef ', func_start + 1)
                    next_class = content.find('\nclass ', func_start + 1)
                    
                    func_end = len(content)
                    if next_def != -1:
                        func_end = min(func_end, next_def)
                    if next_class != -1:
                        func_end = min(func_end, next_class)
                    
                    func_content = content[func_start:func_end]
                    func_lines = len(func_content.split('\n'))
                    
                    if func_lines > 50:
                        disadvantages.append(Disadvantage(
                            category="函数过长",
                            description="存在超过50行的长函数",
                            impact_level="medium",
                            evidence=[f"函数行数: {func_lines}"],
                            mitigation="拆分长函数"
                        ))
                        break
        
        except Exception as e:
            disadvantages.append(Disadvantage(
                category="分析限制",
                description=f"架构分析受限: {e}",
                impact_level="low",
                evidence=["分析错误"],
                mitigation=None
            ))
        
        return disadvantages
    
    async def _analyze_alternatives(self, file_path: str, 
                                  feature_characteristics: List[FeatureCharacteristic]) -> List[Alternative]:
        """分析替代方案"""
        alternatives = []
        
        # 基于特征生成替代方案
        for characteristic in feature_characteristics:
            characteristic_alternatives = await self._generate_alternatives_for_characteristic(characteristic)
            alternatives.extend(characteristic_alternatives)
        
        # 通用替代方案
        general_alternatives = await self._generate_general_alternatives(file_path)
        alternatives.extend(general_alternatives)
        
        return alternatives
    
    async def _generate_alternatives_for_characteristic(self, characteristic: FeatureCharacteristic) -> List[Alternative]:
        """为特征生成替代方案"""
        alternatives = []
        
        if characteristic.category == FeatureCategory.CORE_ENGINE:
            alternatives.append(Alternative(
                name="重构核心引擎",
                description="重新设计核心引擎架构",
                feasibility=0.7,
                cost_estimate="高",
                pros=["提升性能", "降低复杂度", "增强可维护性"],
                cons=["开发周期长", "风险高", "需要充分测试"],
                implementation_effort="高"
            ))
        
        if characteristic.complexity > 0.7:
            alternatives.append(Alternative(
                name="简化模块",
                description="简化复杂模块，拆分为多个小模块",
                feasibility=0.8,
                cost_estimate="中等",
                pros=["降低复杂度", "提高可维护性", "便于测试"],
                cons=["需要重新设计", "可能影响现有功能"],
                implementation_effort="中等"
            ))
        
        return alternatives
    
    async def _generate_general_alternatives(self, file_path: str) -> List[Alternative]:
        """生成通用替代方案"""
        alternatives = []
        
        alternatives.append(Alternative(
            name="保留并优化",
            description="保留现有文件，进行优化改进",
            feasibility=0.9,
            cost_estimate="低",
            pros=["风险低", "保持连续性", "改进现有功能"],
            cons=["可能无法根本解决问题", "技术债务依然存在"],
            implementation_effort="低"
        ))
        
        alternatives.append(Alternative(
            name="完全重写",
            description="完全重写文件功能",
            feasibility=0.6,
            cost_estimate="高",
            pros=["彻底解决问题", "采用最新技术", "优化架构"],
            cons=["开发周期长", "风险高", "需要充分测试"],
            implementation_effort="高"
        ))
        
        alternatives.append(Alternative(
            name="迁移到其他模块",
            description="将功能迁移到其他现有模块",
            feasibility=0.7,
            cost_estimate="中等",
            pros=["减少文件数量", "功能整合", "降低维护成本"],
            cons=["可能增加其他模块复杂度", "需要重构依赖"],
            implementation_effort="中等"
        ))
        
        return alternatives
    
    async def _analyze_dependencies(self, file_path: str) -> Tuple[List[str], List[str]]:
        """分析依赖关系"""
        dependencies = []
        dependents = []
        
        try:
            content = await self._read_file_content(file_path)
            
            # 分析导入依赖
            import_matches = re.findall(r'import\s+(\w+)|from\s+(\w+)', content)
            for match in import_matches:
                dep = match[0] or match[1]
                if dep and not dep.startswith('.'):
                    dependencies.append(dep)
            
            # 简化的依赖者分析（实际需要扫描整个项目）
            # 这里只是示例
            project_files = list(self.project_root.rglob("*.py"))
            for other_file in project_files:
                if str(other_file.relative_to(self.project_root)) != file_path:
                    try:
                        with open(other_file, 'r', encoding='utf-8') as f:
                            other_content = f.read()
                        
                        # 检查其他文件是否导入当前文件
                        current_module = Path(file_path).stem
                        if f"import {current_module}" in other_content or f"from {current_module}" in other_content:
                            dependents.append(str(other_file.relative_to(self.project_root)))
                    except:
                        continue
        
        except Exception as e:
            print(f"⚠️ 依赖分析失败 {file_path}: {e}")
        
        return dependencies, dependents
    
    async def _analyze_integration_points(self, file_path: str) -> List[str]:
        """分析集成点"""
        integration_points = []
        
        try:
            content = await self._read_file_content(file_path)
            
            # 检查API接口
            if re.search(r'def\s+api_|def\s+endpoint|@app\.|@router\.', content):
                integration_points.append("API接口")
            
            # 检查数据库集成
            if any(keyword in content.lower() for keyword in ['database', 'db.', 'sql', 'query']):
                integration_points.append("数据库集成")
            
            # 检查文件系统集成
            if any(keyword in content.lower() for keyword in ['file.', 'open(', 'path.', 'os.']):
                integration_points.append("文件系统集成")
            
            # 检查网络集成
            if any(keyword in content.lower() for keyword in ['http', 'request', 'response', 'socket']):
                integration_points.append("网络集成")
            
            # 检查缓存集成
            if any(keyword in content.lower() for keyword in ['cache', 'redis', 'memcache']):
                integration_points.append("缓存集成")
            
        except Exception as e:
            print(f"⚠️ 集成点分析失败 {file_path}: {e}")
        
        return integration_points
    
    async def _generate_retention_justification(self, file_path: str,
                                             feature_characteristics: List[FeatureCharacteristic],
                                             advantages: List[Advantage],
                                             disadvantages: List[Disadvantage]) -> str:
        """生成保留理由"""
        justification_parts = []
        
        justification_parts.append(f"## 保留 {file_path} 的理由")
        justification_parts.append("")
        
        # 功能价值
        if feature_characteristics:
            justification_parts.append("### 功能价值")
            for characteristic in feature_characteristics:
                if characteristic.value_level in [ValueLevel.CRITICAL, ValueLevel.HIGH]:
                    justification_parts.append(f"- **{characteristic.name}**: {characteristic.description}")
                    justification_parts.append(f"  - 价值等级: {characteristic.value_level.value}")
                    justification_parts.append(f"  - 独特性: {characteristic.uniqueness:.1%}")
            justification_parts.append("")
        
        # 优势分析
        if advantages:
            justification_parts.append("### 主要优势")
            high_impact_advantages = [adv for adv in advantages if adv.impact_level in ["high", "medium"]]
            for advantage in high_impact_advantages:
                justification_parts.append(f"- **{advantage.category}**: {advantage.description}")
                if advantage.quantification:
                    justification_parts.append(f"  - 量化指标: {advantage.quantification}")
            justification_parts.append("")
        
        # 依赖关系
        try:
            dependencies, dependents = await self._analyze_dependencies(file_path)
            if dependents:
                justification_parts.append("### 依赖关系")
                justification_parts.append(f"- 被 {len(dependents)} 个其他模块依赖:")
                for dependent in dependents[:5]:  # 限制显示数量
                    justification_parts.append(f"  - {dependent}")
                justification_parts.append("")
        except:
            pass
        
        # 集成点
        try:
            integration_points = await self._analyze_integration_points(file_path)
            if integration_points:
                justification_parts.append("### 集成点")
                for point in integration_points:
                    justification_parts.append(f"- {point}")
                justification_parts.append("")
        except:
            pass
        
        # 结论
        justification_parts.append("### 结论")
        if len(advantages) > len(disadvantages):
            justification_parts.append("基于以上分析，该文件的优势明显大于劣势，建议保留。")
        elif any(char.value_level == ValueLevel.CRITICAL for char in feature_characteristics):
            justification_parts.append("该文件包含关键功能，虽然存在一些问题，但建议保留并进行优化。")
        else:
            justification_parts.append("该文件具有一定的价值，建议保留但需要持续改进。")
        
        return "\n".join(justification_parts)
    
    async def _generate_removal_justification(self, file_path: str,
                                            feature_characteristics: List[FeatureCharacteristic],
                                            disadvantages: List[Disadvantage]) -> Optional[str]:
        """生成删除理由"""
        # 只有在充分理由时才生成删除理由
        removal_reasons = []
        
        # 检查是否有充分的删除理由
        high_impact_disadvantages = [dis for dis in disadvantages if dis.impact_level == "high"]
        low_value_features = [char for char in feature_characteristics if char.value_level in [ValueLevel.LOW, ValueLevel.NEGLIGIBLE]]
        
        if not high_impact_disadvantages and not low_value_features:
            return None
        
        justification_parts = []
        justification_parts.append(f"## 删除 {file_path} 的理由")
        justification_parts.append("")
        
        # 严重问题
        if high_impact_disadvantages:
            justification_parts.append("### 严重问题")
            for disadvantage in high_impact_disadvantages:
                justification_parts.append(f"- **{disadvantage.category}**: {disadvantage.description}")
                if disadvantage.mitigation:
                    justification_parts.append(f"  - 缓解方案: {disadvantage.mitigation}")
            justification_parts.append("")
        
        # 低价值特征
        if low_value_features:
            justification_parts.append("### 低价值特征")
            for characteristic in low_value_features:
                justification_parts.append(f"- **{characteristic.name}**: {characteristic.description}")
                justification_parts.append(f"  - 价值等级: {characteristic.value_level.value}")
                justification_parts.append(f"  - 技术债务: {characteristic.technical_debt:.1%}")
            justification_parts.append("")
        
        # 替代方案
        try:
            alternatives = await self._generate_general_alternatives(file_path)
            if alternatives:
                justification_parts.append("### 替代方案")
                for alternative in alternatives:
                    justification_parts.append(f"- **{alternative.name}**: {alternative.description}")
                    justification_parts.append(f"  - 可行性: {alternative.feasibility:.1%}")
                    justification_parts.append(f"  - 实施难度: {alternative.implementation_effort}")
                justification_parts.append("")
        except:
            pass
        
        # 结论
        justification_parts.append("### 结论")
        justification_parts.append("基于以上分析，该文件存在严重问题且价值较低，建议删除。")
        
        return "\n".join(justification_parts)
    
    async def _analyze_replacement_options(self, file_path: str, alternatives: List[Alternative]) -> List[str]:
        """分析替换选项"""
        options = []
        
        for alternative in alternatives:
            if alternative.feasibility > 0.6:
                options.append(f"{alternative.name}: {alternative.description}")
        
        return options
    
    async def _perform_overall_assessment(self, feature_characteristics: List[FeatureCharacteristic],
                                         advantages: List[Advantage],
                                         disadvantages: List[Disadvantage]) -> str:
        """执行整体评估"""
        assessment_parts = []
        
        # 计算综合评分
        high_value_features = len([char for char in feature_characteristics if char.value_level in [ValueLevel.CRITICAL, ValueLevel.HIGH]])
        high_impact_advantages = len([adv for adv in advantages if adv.impact_level == "high"])
        high_impact_disadvantages = len([dis for dis in disadvantages if dis.impact_level == "high"])
        
        # 评估结论
        if high_value_features > 0 or high_impact_advantages > high_impact_disadvantages:
            assessment = "该文件具有较高的价值和重要性，建议保留。"
        elif high_impact_disadvantages > high_impact_advantages:
            assessment = "该文件存在较多严重问题，建议考虑删除或重构。"
        else:
            assessment = "该文件价值一般，需要根据具体情况决定保留或删除。"
        
        assessment_parts.append("## 整体评估")
        assessment_parts.append("")
        assessment_parts.append(f"**评估结论**: {assessment}")
        assessment_parts.append("")
        assessment_parts.append(f"**统计信息**:")
        assessment_parts.append(f"- 高价值特征: {high_value_features}个")
        assessment_parts.append(f"- 高影响优势: {high_impact_advantages}个")
        assessment_parts.append(f"- 高影响劣势: {high_impact_disadvantages}个")
        assessment_parts.append("")
        
        return "\n".join(assessment_parts)
    
    async def _generate_recommendation(self, file_path: str, 
                                     overall_assessment: str,
                                     alternatives: List[Alternative]) -> str:
        """生成推荐建议"""
        recommendation_parts = []
        
        recommendation_parts.append("## 推荐建议")
        recommendation_parts.append("")
        
        # 基于整体评估生成推荐
        if "建议保留" in overall_assessment:
            recommendation_parts.append("### 主要建议")
            recommendation_parts.append("1. **保留文件** - 继续维护和使用该文件")
            recommendation_parts.append("2. **优化改进** - 针对识别的问题进行优化")
            recommendation_parts.append("3. **监控评估** - 定期评估文件价值和使用情况")
        elif "建议考虑删除" in overall_assessment:
            recommendation_parts.append("### 主要建议")
            recommendation_parts.append("1. **谨慎删除** - 在充分测试后考虑删除")
            recommendation_parts.append("2. **功能迁移** - 将有用功能迁移到其他模块")
            recommendation_parts.append("3. **备份保留** - 删除前备份以防需要恢复")
        else:
            recommendation_parts.append("### 主要建议")
            recommendation_parts.append("1. **进一步分析** - 收集更多使用数据和反馈")
            recommendation_parts.append("2. **试点测试** - 在小范围内测试替代方案")
            recommendation_parts.append("3. **团队讨论** - 与团队讨论决定最终方案")
        
        recommendation_parts.append("")
        
        # 实施建议
        if alternatives:
            best_alternative = max(alternatives, key=lambda x: x.feasibility)
            recommendation_parts.append("### 实施建议")
            recommendation_parts.append(f"推荐采用: **{best_alternative.name}**")
            recommendation_parts.append(f"理由: {best_alternative.description}")
            recommendation_parts.append(f"可行性: {best_alternative.feasibility:.1%}")
            recommendation_parts.append(f"实施难度: {best_alternative.implementation_effort}")
            recommendation_parts.append("")
        
        return "\n".join(recommendation_parts)
    
    async def _read_file_content(self, file_path: str) -> str:
        """读取文件内容"""
        full_path = self.project_root / file_path
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"⚠️ 读取文件失败 {file_path}: {e}")
            return ""
    
    def _load_feature_patterns(self) -> Dict[str, Any]:
        """加载功能模式"""
        return {
            "arq_engine": {
                "name": "ARQ引擎",
                "description": "自适应推理查询引擎",
                "category": "core_engine",
                "value_level": "critical",
                "filename_patterns": [r".*arq.*engine.*", r".*adaptive.*reasoning.*"],
                "content_patterns": [r"class.*ARQ", r"def.*reasoning"],
                "keywords": ["reasoning", "query", "adaptive", "intelligent"],
                "usage_frequency": "high",
                "user_impact": "high",
                "business_value": "critical",
                "technical_debt": 0.3
            },
            "hrrk_kernel": {
                "name": "HRRK内核",
                "description": "高性能推理内核",
                "category": "core_engine",
                "value_level": "critical",
                "filename_patterns": [r".*hrrk.*kernel.*", r".*high.*performance.*"],
                "content_patterns": [r"class.*HRRK", r"def.*kernel"],
                "keywords": ["kernel", "performance", "high-speed", "reasoning"],
                "usage_frequency": "high",
                "user_impact": "high",
                "business_value": "critical",
                "technical_debt": 0.2
            },
            "refrag_system": {
                "name": "REFRAG系统",
                "description": "检索增强生成系统",
                "category": "core_engine",
                "value_level": "high",
                "filename_patterns": [r".*refrag.*", r".*retrieval.*"],
                "content_patterns": [r"class.*REFRAG", r"def.*retrieval"],
                "keywords": ["retrieval", "generation", "frag", "search"],
                "usage_frequency": "high",
                "user_impact": "high",
                "business_value": "high",
                "technical_debt": 0.3
            },
            "cache_system": {
                "name": "缓存系统",
                "description": "智能缓存管理",
                "category": "performance_module",
                "value_level": "high",
                "filename_patterns": [r".*cache.*", r".*caching.*"],
                "content_patterns": [r"class.*Cache", r"def.*cache"],
                "keywords": ["cache", "caching", "memory", "performance"],
                "usage_frequency": "high",
                "user_impact": "medium",
                "business_value": "high",
                "technical_debt": 0.2
            },
            "workflow_engine": {
                "name": "工作流引擎",
                "description": "业务流程管理",
                "category": "workflow_system",
                "value_level": "high",
                "filename_patterns": [r".*workflow.*", r".*process.*"],
                "content_patterns": [r"class.*Workflow", r"def.*workflow"],
                "keywords": ["workflow", "process", "flow", "orchestration"],
                "usage_frequency": "medium",
                "user_impact": "medium",
                "business_value": "high",
                "technical_debt": 0.3
            },
            "security_module": {
                "name": "安全模块",
                "description": "系统安全防护",
                "category": "security_module",
                "value_level": "high",
                "filename_patterns": [r".*security.*", r".*auth.*"],
                "content_patterns": [r"class.*Security", r"def.*security"],
                "keywords": ["security", "authentication", "authorization", "protection"],
                "usage_frequency": "medium",
                "user_impact": "high",
                "business_value": "critical",
                "technical_debt": 0.2
            },
            "test_module": {
                "name": "测试模块",
                "description": "自动化测试",
                "category": "test_module",
                "value_level": "medium",
                "filename_patterns": [r".*test.*", r".*spec.*"],
                "content_patterns": [r"def test_", r"class.*Test"],
                "keywords": ["test", "testing", "spec", "assert"],
                "usage_frequency": "medium",
                "user_impact": "low",
                "business_value": "medium",
                "technical_debt": 0.4
            },
            "utility_module": {
                "name": "工具模块",
                "description": "通用工具函数",
                "category": "utility_module",
                "value_level": "medium",
                "filename_patterns": [r".*util.*", r".*helper.*", r".*tool.*"],
                "content_patterns": [r"def.*util", r"def.*helper"],
                "keywords": ["utility", "helper", "tool", "common"],
                "usage_frequency": "medium",
                "user_impact": "low",
                "business_value": "medium",
                "technical_debt": 0.3
            }
        }
    
    def _load_value_criteria(self) -> Dict[str, Any]:
        """加载价值评估标准"""
        return {
            "critical": {
                "description": "关键功能，系统核心组件",
                "impact": "删除会导致系统无法正常运行",
                "usage_threshold": "> 80% 使用频率",
                "business_impact": "直接影响核心业务"
            },
            "high": {
                "description": "重要功能，显著提升用户体验",
                "impact": "删除会严重影响系统功能",
                "usage_threshold": "50-80% 使用频率",
                "business_impact": "影响重要业务流程"
            },
            "medium": {
                "description": "有用功能，提供增值服务",
                "impact": "删除会影响部分用户体验",
                "usage_threshold": "20-50% 使用频率",
                "business_impact": "影响辅助业务功能"
            },
            "low": {
                "description": "次要功能，使用较少",
                "impact": "删除影响有限",
                "usage_threshold": "5-20% 使用频率",
                "business_impact": "影响边缘业务功能"
            },
            "negligible": {
                "description": "几乎不使用的功能",
                "impact": "删除几乎无影响",
                "usage_threshold": "< 5% 使用频率",
                "business_impact": "几乎无业务影响"
            }
        }

# 使用示例
async def main():
    """主函数"""
    project_root = "."
    
    analyzer = FeatureAnalysisModule(project_root)
    
    # 示例：分析一个文件
    file_path = "example_module.py"
    analysis = await analyzer.analyze_comprehensive_features(file_path)
    
    print(f"🎉 功能分析完成!")
    print(f"📊 文件: {analysis.file_path}")
    print(f"🎯 特征数量: {len(analysis.feature_characteristics)}")
    print(f"💪 优势数量: {len(analysis.advantages)}")
    print(f"⚠️ 劣势数量: {len(analysis.disadvantages)}")
    print(f"🔄 替代方案: {len(analysis.alternatives)}")
    print(f"💡 推荐: {analysis.recommendation}")

if __name__ == "__main__":
    asyncio.run(main())