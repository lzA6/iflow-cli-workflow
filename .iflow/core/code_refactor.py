#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 代码重构工具 (Code Refactoring Tools)
=======================================

提供代码重构和规范化功能：
- 重复代码检测
- 命名规范统一
- 代码结构优化
- 自动重构建议
- 代码质量分析

特性：
- 智能重复代码检测
- 命名规范检查和修复
- 代码复杂度分析
- 重构建议生成

作者: iFlow代码质量团队
版本: 1.0.0
日期: 2025-11-16
"""

import os
import ast
import re
import json
import difflib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class CodeIssue:
    """代码问题"""
    file_path: str
    line_number: int
    issue_type: str
    description: str
    severity: str  # low, medium, high, critical
    suggestion: str
    code_snippet: str

@dataclass
class DuplicateCodeBlock:
    """重复代码块"""
    files: List[Tuple[str, int, int]]  # (file_path, start_line, end_line)
    similarity: float
    code_hash: str
    line_count: int

@dataclass
class NamingIssue:
    """命名问题"""
    file_path: str
    line_number: int
    current_name: str
    issue_type: str
    suggestion: str
    severity: str

class CodeAnalyzer:
    """代码分析器"""
    
    def __init__(self):
        self.issues: List[CodeIssue] = []
        self.duplicates: List[DuplicateCodeBlock] = []
        self.naming_issues: List[NamingIssue] = []
        
        # 命名规范
        self.naming_patterns = {
            'variable': re.compile(r'^[a-z_][a-z0-9_]*$'),  # snake_case
            'function': re.compile(r'^[a-z_][a-z0-9_]*$'),  # snake_case
            'class': re.compile(r'^[A-Z][a-zA-Z0-9]*$'),  # PascalCase
            'constant': re.compile(r'^[A-Z_][A-Z0-9_]*$'),  # UPPER_CASE
            'private': re.compile(r'^_[a-z_][a-z0-9_]*$'),  # _snake_case
            'dunder': re.compile(r'^__[a-z_][a-z0-9_]*__$'),  # __snake_case__
        }
    
    def analyze_directory(self, directory: str, patterns: List[str] = None) -> Dict[str, Any]:
        """分析目录中的代码"""
        if patterns is None:
            patterns = ['*.py']
        
        python_files = []
        for pattern in patterns:
            python_files.extend(Path(directory).rglob(pattern))
        
        logger.info(f"分析 {len(python_files)} 个Python文件")
        
        # 分析每个文件
        for file_path in python_files:
            self.analyze_file(str(file_path))
        
        # 检测重复代码
        self.detect_duplicates(python_files)
        
        return {
            'issues': len(self.issues),
            'duplicates': len(self.duplicates),
            'naming_issues': len(self.naming_issues),
            'files_analyzed': len(python_files)
        }
    
    def analyze_file(self, file_path: str):
        """分析单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析AST
            tree = ast.parse(content)
            
            # 分析命名规范
            self._analyze_naming(tree, file_path, content)
            
            # 分析代码结构
            self._analyze_structure(tree, file_path, content)
            
        except Exception as e:
            logger.error(f"分析文件失败 {file_path}: {e}")
    
    def _analyze_naming(self, tree: ast.AST, file_path: str, content: str):
        """分析命名规范"""
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._check_function_naming(node, file_path, lines)
            elif isinstance(node, ast.ClassDef):
                self._check_class_naming(node, file_path, lines)
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    self._check_variable_naming(node, file_path, lines)
    
    def _check_function_naming(self, node: ast.FunctionDef, file_path: str, lines: List[str]):
        """检查函数命名"""
        name = node.name
        
        # 检查是否符合规范
        if not self.naming_patterns['function'].match(name):
            severity = 'high' if name.isupper() else 'medium'
            
            suggestion = self._suggest_function_name(name)
            
            self.naming_issues.append(NamingIssue(
                file_path=file_path,
                line_number=node.lineno,
                current_name=name,
                issue_type='function_naming',
                suggestion=suggestion,
                severity=severity
            ))
    
    def _check_class_naming(self, node: ast.ClassDef, file_path: str, lines: List[str]):
        """检查类命名"""
        name = node.name
        
        if not self.naming_patterns['class'].match(name):
            severity = 'high'
            
            suggestion = self._suggest_class_name(name)
            
            self.naming_issues.append(NamingIssue(
                file_path=file_path,
                line_number=node.lineno,
                current_name=name,
                issue_type='class_naming',
                suggestion=suggestion,
                severity=severity
            ))
    
    def _check_variable_naming(self, node: ast.Name, file_path: str, lines: List[str]):
        """检查变量命名"""
        name = node.id
        
        # 跳过特殊变量
        if name.startswith('__') and name.endswith('__'):
            return
        
        # 检查是否是常量
        if name.isupper():
            if not self.naming_patterns['constant'].match(name):
                suggestion = self._suggest_constant_name(name)
                self.naming_issues.append(NamingIssue(
                    file_path=file_path,
                    line_number=node.lineno,
                    current_name=name,
                    issue_type='constant_naming',
                    suggestion=suggestion,
                    severity='medium'
                ))
        else:
            # 普通变量
            if not self.naming_patterns['variable'].match(name):
                suggestion = self._suggest_variable_name(name)
                self.naming_issues.append(NamingIssue(
                    file_path=file_path,
                    line_number=node.lineno,
                    current_name=name,
                    issue_type='variable_naming',
                    suggestion=suggestion,
                    severity='low'
                ))
    
    def _suggest_function_name(self, name: str) -> str:
        """建议函数名"""
        # 转换为snake_case
        suggested = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        return suggested if suggested != name else f"{name.lower()}_function"
    
    def _suggest_class_name(self, name: str) -> str:
        """建议类名"""
        # 转换为PascalCase
        suggested = ''.join(word.capitalize() for word in name.split('_'))
        return suggested if suggested != name else f"{name.capitalize()}Class"
    
    def _suggest_variable_name(self, name: str) -> str:
        """建议变量名"""
        if name.isupper():
            return name.lower()
        elif name[0].isupper():
            return name[0].lower() + name[1:]
        else:
            return f"{name}_var"
    
    def _suggest_constant_name(self, name: str) -> str:
        """建议常量名"""
        return name.upper()
    
    def _analyze_structure(self, tree: ast.AST, file_path: str, content: str):
        """分析代码结构"""
        lines = content.split('\n')
        
        # 检查函数复杂度
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._check_function_complexity(node, file_path, lines)
            elif isinstance(node, ast.ClassDef):
                self._check_class_complexity(node, file_path, lines)
    
    def _check_function_complexity(self, node: ast.FunctionDef, file_path: str, lines: List[str]):
        """检查函数复杂度"""
        # 计算圈复杂度
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        
        # 检查函数长度
        func_lines = node.end_lineno - node.lineno + 1
        
        if complexity > 10:
            severity = 'high' if complexity > 20 else 'medium'
            self.issues.append(CodeIssue(
                file_path=file_path,
                line_number=node.lineno,
                issue_type='high_complexity',
                description=f"函数 '{node.name}' 圈复杂度过高: {complexity}",
                severity=severity,
                suggestion="考虑拆分函数或简化逻辑",
                code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else ""
            ))
        
        if func_lines > 50:
            self.issues.append(CodeIssue(
                file_path=file_path,
                line_number=node.lineno,
                issue_type='long_function',
                description=f"函数 '{node.name}' 过长: {func_lines} 行",
                severity='medium',
                suggestion="考虑拆分为更小的函数",
                code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else ""
            ))
    
    def _check_class_complexity(self, node: ast.ClassDef, file_path: str, lines: List[str]):
        """检查类复杂度"""
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        
        if len(methods) > 20:
            self.issues.append(CodeIssue(
                file_path=file_path,
                line_number=node.lineno,
                issue_type='large_class',
                description=f"类 '{node.name}' 方法过多: {len(methods)}",
                severity='medium',
                suggestion="考虑拆分为多个类或使用组合模式",
                code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else ""
            ))
    
    def detect_duplicates(self, files: List[Path], min_lines: int = 5):
        """检测重复代码"""
        code_blocks = {}
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                # 提取代码块（函数、类等）
                for i, line in enumerate(lines):
                    if len(line.strip()) >= min_lines:
                        # 简单的哈希（实际中可以使用更复杂的算法）
                        block_hash = hash(line.strip())
                        
                        if block_hash not in code_blocks:
                            code_blocks[block_hash] = []
                        
                        code_blocks[block_hash].append((str(file_path), i + 1, i + 1))
                        
            except Exception as e:
                logger.error(f"检测重复代码失败 {file_path}: {e}")
        
        # 找出重复的代码块
        for block_hash, occurrences in code_blocks.items():
            if len(occurrences) > 1:
                self.duplicates.append(DuplicateCodeBlock(
                    files=occurrences,
                    similarity=1.0,  # 简化处理
                    code_hash=str(block_hash),
                    line_count=1
                ))

class CodeRefactor:
    """代码重构器"""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.refactoring_rules = {
            'extract_function': self._extract_function,
            'rename_variable': self._rename_variable,
            'simplify_condition': self._simplify_condition,
            'remove_duplicates': self._remove_duplicates
        }
    
    def refactor_directory(self, directory: str, auto_fix: bool = False) -> Dict[str, Any]:
        """重构目录中的代码"""
        # 分析代码
        analysis_result = self.analyzer.analyze_directory(directory)
        
        refactoring_plan = {
            'analysis': analysis_result,
            'fixes': [],
            'auto_fixes': []
        }
        
        # 生成修复建议
        for issue in self.analyzer.issues:
            fix = self._generate_fix_suggestion(issue)
            refactoring_plan['fixes'].append(fix)
        
        for naming_issue in self.analyzer.naming_issues:
            fix = self._generate_naming_fix(naming_issue)
            refactoring_plan['fixes'].append(fix)
        
        for duplicate in self.analyzer.duplicates:
            fix = self._generate_duplicate_fix(duplicate)
            refactoring_plan['fixes'].append(fix)
        
        # 自动修复
        if auto_fix:
            auto_fixes = self._apply_auto_fixes(directory)
            refactoring_plan['auto_fixes'] = auto_fixes
        
        return refactoring_plan
    
    def _generate_fix_suggestion(self, issue: CodeIssue) -> Dict[str, Any]:
        """生成修复建议"""
        return {
            'type': 'issue_fix',
            'file_path': issue.file_path,
            'line_number': issue.line_number,
            'issue_type': issue.issue_type,
            'description': issue.description,
            'suggestion': issue.suggestion,
            'severity': issue.severity,
            'auto_fixable': issue.issue_type in ['long_function', 'large_class']
        }
    
    def _generate_naming_fix(self, naming_issue: NamingIssue) -> Dict[str, Any]:
        """生成命名修复建议"""
        return {
            'type': 'naming_fix',
            'file_path': naming_issue.file_path,
            'line_number': naming_issue.line_number,
            'current_name': naming_issue.current_name,
            'suggested_name': naming_issue.suggestion,
            'issue_type': naming_issue.issue_type,
            'severity': naming_issue.severity,
            'auto_fixable': naming_issue.severity in ['low', 'medium']
        }
    
    def _generate_duplicate_fix(self, duplicate: DuplicateCodeBlock) -> Dict[str, Any]:
        """生成重复代码修复建议"""
        return {
            'type': 'duplicate_fix',
            'files': duplicate.files,
            'similarity': duplicate.similarity,
            'suggestion': "提取公共函数或使用继承",
            'auto_fixable': False
        }
    
    def _apply_auto_fixes(self, directory: str) -> List[Dict[str, Any]]:
        """应用自动修复"""
        auto_fixes = []
        
        # 这里可以实现具体的自动修复逻辑
        # 例如：自动重命名变量、提取函数等
        
        return auto_fixes
    
    def _extract_function(self, file_path: str, start_line: int, end_line: int):
        """提取函数"""
        # 实现函数提取逻辑
        pass
    
    def _rename_variable(self, file_path: str, old_name: str, new_name: str):
        """重命名变量"""
        # 实现变量重命名逻辑
        pass
    
    def _simplify_condition(self, file_path: str, line_number: int):
        """简化条件"""
        # 实现条件简化逻辑
        pass
    
    def _remove_duplicates(self, duplicate: DuplicateCodeBlock):
        """移除重复代码"""
        # 实现重复代码移除逻辑
        pass

class NamingStandardizer:
    """命名规范化器"""
    
    def __init__(self):
        self.conversion_rules = {
            'camel_to_snake': self._camel_to_snake,
            'snake_to_camel': self._snake_to_camel,
            'snake_to_pascal': self._snake_to_pascal,
            'normalize': self._normalize_name
        }
    
    def standardize_name(self, name: str, target_style: str) -> str:
        """标准化名称"""
        if target_style in self.conversion_rules:
            return self.conversion_rules[target_style](name)
        return name
    
    def _camel_to_snake(self, name: str) -> str:
        """驼峰转下划线"""
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    
    def _snake_to_camel(self, name: str) -> str:
        """下划线转驼峰"""
        components = name.split('_')
        return components[0] + ''.join(word.capitalize() for word in components[1:])
    
    def _snake_to_pascal(self, name: str) -> str:
        """下划线转帕斯卡"""
        return ''.join(word.capitalize() for word in name.split('_'))
    
    def _normalize_name(self, name: str) -> str:
        """标准化名称"""
        # 移除特殊字符，转换为下划线
        normalized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # 移除多余的下划线
        normalized = re.sub(r'_+', '_', normalized)
        # 移除开头和结尾的下划线
        normalized = normalized.strip('_')
        return normalized.lower()

def create_refactoring_report(directory: str) -> str:
    """创建重构报告"""
    refactor = CodeRefactor()
    result = refactor.refactor_directory(directory)
    
    report = f"""
# 代码重构报告

## 分析概览
- 分析文件数: {result['analysis']['files_analyzed']}
- 发现问题数: {result['analysis']['issues']}
- 重复代码块: {result['analysis']['duplicates']}
- 命名问题: {result['analysis']['naming_issues']}

## 修复建议
"""
    
    for fix in result['fixes']:
        report += f"""
### {fix['issue_type'].replace('_', ' ').title()}
- **文件**: {fix['file_path']}
- **行号**: {fix.get('line_number', 'N/A')}
- **严重程度**: {fix['severity']}
- **描述**: {fix.get('description', fix.get('current_name', 'N/A'))}
- **建议**: {fix['suggestion']}
- **可自动修复**: {'是' if fix.get('auto_fixable') else '否'}

"""
    
    return report

if __name__ == "__main__":
    # 测试代码重构工具
    print("🔧 测试代码重构工具")
    
    # 分析当前目录
    analyzer = CodeAnalyzer()
    result = analyzer.analyze_directory(".", ["*.py"])
    
    print(f"分析结果: {result}")
    
    # 生成重构报告
    report = create_refactoring_report(".")
    
    # 保存报告
    report_file = "code_refactoring_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 重构报告已保存: {report_file}")