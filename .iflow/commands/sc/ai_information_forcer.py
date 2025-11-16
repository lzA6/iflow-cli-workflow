#!/usr/bin/env python3
"""
AI强制信息传递系统
确保AI完全理解项目状态和测试要求
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import hashlib

class AIInformationForcer:
    """AI信息强制传递器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.ai_context_dir = self.project_root / ".iflow" / "temp_docs" / "ai_context"
        self.ai_context_dir.mkdir(parents=True, exist_ok=True)
        
    async def force_ai_awareness(self, test_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """强制AI信息传递"""
        print("🤖 启动AI强制信息传递系统...")
        
        # 1. 创建完整项目上下文
        project_context = await self._create_project_context()
        
        # 2. 创建测试要求上下文
        test_requirements = await self._create_test_requirements()
        
        # 3. 创建决策依据上下文
        decision_context = await self._create_decision_context()
        
        # 4. 创建功能分析上下文
        functionality_context = await self._create_functionality_context()
        
        # 5. 保存所有上下文
        context_files = await self._save_contexts({
            "project_context": project_context,
            "test_requirements": test_requirements,
            "decision_context": decision_context,
            "functionality_context": functionality_context
        })
        
        # 6. 创建强制传递指令
        force_commands = await self._create_force_commands(context_files)
        
        # 7. 生成AI理解验证
        verification = await self._create_ai_verification()
        
        print("✅ AI强制信息传递完成")
        print(f"📁 上下文文件已保存到: {self.ai_context_dir}")
        
        return {
            "status": "success",
            "context_files": context_files,
            "force_commands": force_commands,
            "verification": verification
        }
    
    async def _create_project_context(self) -> Dict[str, Any]:
        """创建项目上下文"""
        print("📋 创建项目上下文...")
        
        context = {
            "project_name": "iFlow CLI V16 Quantum Evolution",
            "project_description": "企业级智能CLI工具，集成ARQ引擎、HRRK内核、REFRAG系统等核心组件",
            "project_root": str(self.project_root),
            "timestamp": datetime.now().isoformat(),
            "project_structure": await self._get_project_structure(),
            "core_modules": await self._get_core_modules(),
            "dependencies": await self._get_dependencies(),
            "configuration": await self._get_configuration(),
            "recent_changes": await self._get_recent_changes(),
            "critical_files": await self._get_critical_files()
        }
        
        return context
    
    async def _create_test_requirements(self) -> Dict[str, Any]:
        """创建测试要求上下文"""
        print("🧪 创建测试要求上下文...")
        
        requirements = {
            "test_objectives": [
                {
                    "objective": "全面测试覆盖分析",
                    "description": "确保所有核心模块都有充分的测试覆盖",
                    "success_criteria": "覆盖率 >= 25%",
                    "priority": "high"
                },
                {
                    "objective": "深度代码质量审查",
                    "description": "检查代码质量、复杂度、重复代码等问题",
                    "success_criteria": "无严重代码质量问题",
                    "priority": "high"
                },
                {
                    "objective": "安全性漏洞扫描",
                    "description": "识别潜在的安全风险和漏洞",
                    "success_criteria": "无高危安全问题",
                    "priority": "critical"
                },
                {
                    "objective": "性能基准测试",
                    "description": "评估系统性能和资源使用情况",
                    "success_criteria": "性能指标在可接受范围内",
                    "priority": "medium"
                },
                {
                    "objective": "项目结构优化分析",
                    "description": "分析项目结构合理性，提出优化建议",
                    "success_criteria": "结构清晰，无冗余文件",
                    "priority": "medium"
                }
            ],
            "mandatory_requirements": [
                "每一步都必须提供完整依据和解释",
                "所有文件决策都需要详细推理过程",
                "功能特点和优缺点必须明确列出",
                "删除文件必须有充分理由和证据",
                "保留文件需要说明其独特价值和不可替代性",
                "必须提供自我反省和推理过程",
                "必须对比分析删除前后的影响"
            ],
            "decision_framework": {
                "file_retention_criteria": [
                    "核心功能模块",
                    "无重复实现",
                    "性能关键路径",
                    "安全关键组件",
                    "用户直接接口",
                    "配置和设置文件",
                    "文档和说明文件"
                ],
                "file_removal_criteria": [
                    "功能完全重复",
                    "无实际用途",
                    "过时版本",
                    "测试用临时文件",
                    "调试代码",
                    "冗余依赖"
                ],
                "analysis_requirements": [
                    "功能完整性分析",
                    "性能影响评估",
                    "依赖关系分析",
                    "安全性评估",
                    "维护成本分析",
                    "用户体验影响"
                ]
            }
        }
        
        return requirements
    
    async def _create_decision_context(self) -> Dict[str, Any]:
        """创建决策依据上下文"""
        print("⚖️ 创建决策依据上下文...")
        
        context = {
            "decision_principles": [
                {
                    "principle": "证据驱动决策",
                    "description": "所有决策必须基于具体的证据和数据",
                    "application": "文件分析、性能测试、用户反馈"
                },
                {
                    "principle": "影响最小化",
                    "description": "确保决策对系统的影响最小化",
                    "application": "向后兼容性、API稳定性"
                },
                {
                    "principle": "价值最大化",
                    "description": "确保每个组件都为用户提供最大价值",
                    "application": "功能必要性、性能提升"
                }
            ],
            "analysis_templates": {
                "file_analysis": {
                    "required_fields": [
                        "文件路径和大小",
                        "最后修改时间",
                        "功能描述",
                        "依赖关系",
                        "调用关系",
                        "性能指标",
                        "安全性评估",
                        "维护复杂度"
                    ],
                    "decision_factors": [
                        "功能独特性",
                        "性能贡献",
                        "安全重要性",
                        "用户体验影响",
                        "维护成本",
                        "未来发展规划"
                    ]
                },
                "retention_justification": {
                    "structure": [
                        "功能概述",
                        "独特价值分析",
                        "替代方案对比",
                        "删除影响评估",
                        "保留理由总结"
                    ],
                    "evidence_required": [
                        "代码分析结果",
                        "性能测试数据",
                        "依赖关系图",
                        "用户使用统计",
                        "安全评估报告"
                    ]
                }
            }
        }
        
        return context
    
    async def _create_functionality_context(self) -> Dict[str, Any]:
        """创建功能分析上下文"""
        print("🔍 创建功能分析上下文...")
        
        context = {
            "functionality_categories": {
                "core_engine": {
                    "description": "核心引擎模块",
                    "examples": ["ARQ引擎", "HRRK内核", "REFRAG系统"],
                    "characteristics": ["高性能", "核心功能", "复杂算法"],
                    "retention_priority": "critical",
                    "analysis_focus": ["性能", "稳定性", "安全性"]
                },
                "workflow_system": {
                    "description": "工作流系统",
                    "examples": ["工作流引擎", "任务调度器", "状态管理器"],
                    "characteristics": ["流程控制", "状态管理", "任务协调"],
                    "retention_priority": "high",
                    "analysis_focus": ["可靠性", "扩展性", "易用性"]
                },
                "knowledge_base": {
                    "description": "知识库系统",
                    "examples": ["知识库管理器", "向量存储", "搜索引擎"],
                    "characteristics": ["数据存储", "检索功能", "智能分析"],
                    "retention_priority": "high",
                    "analysis_focus": ["数据完整性", "检索效率", "智能程度"]
                },
                "utility_modules": {
                    "description": "工具模块",
                    "examples": ["缓存系统", "错误处理器", "日志系统"],
                    "characteristics": ["辅助功能", "性能优化", "系统支持"],
                    "retention_priority": "medium",
                    "analysis_focus": ["性能提升", "稳定性", "维护成本"]
                },
                "test_modules": {
                    "description": "测试模块",
                    "examples": ["单元测试", "集成测试", "性能测试"],
                    "characteristics": ["质量保证", "回归测试", "自动化"],
                    "retention_priority": "medium",
                    "analysis_focus": ["覆盖率", "有效性", "维护性"]
                }
            },
            "analysis_checklist": {
                "functionality_assessment": [
                    "主要功能是什么？",
                    "解决了什么问题？",
                    "用户价值是什么？",
                    "使用频率如何？",
                    "是否有替代方案？"
                ],
                "technical_analysis": [
                    "代码复杂度如何？",
                    "性能表现如何？",
                    "依赖关系复杂吗？",
                    "安全性如何？",
                    "维护成本高吗？"
                ],
                "business_value": [
                    "对用户的价值是什么？",
                    "对业务的重要性如何？",
                    "竞争优势在哪里？",
                    "未来发展潜力如何？",
                    "风险影响程度如何？"
                ]
            }
        }
        
        return context
    
    async def _get_project_structure(self) -> Dict[str, Any]:
        """获取项目结构"""
        structure = {
            "directories": {},
            "files": {},
            "statistics": {}
        }
        
        total_files = 0
        total_dirs = 0
        python_files = 0
        test_files = 0
        
        for root, dirs, files in os.walk(self.project_root):
            # 跳过隐藏目录和缓存
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            rel_root = os.path.relpath(root, self.project_root)
            if rel_root == '.':
                rel_root = 'root'
            
            structure["directories"][rel_root] = dirs
            structure["files"][rel_root] = files
            
            total_dirs += len(dirs)
            total_files += len(files)
            
            for file in files:
                if file.endswith('.py'):
                    python_files += 1
                if 'test' in file.lower():
                    test_files += 1
        
        structure["statistics"] = {
            "total_files": total_files,
            "total_directories": total_dirs,
            "python_files": python_files,
            "test_files": test_files
        }
        
        return structure
    
    async def _get_core_modules(self) -> List[Dict[str, Any]]:
        """获取核心模块信息"""
        core_dir = self.project_root / ".iflow" / "core"
        modules = []
        
        if core_dir.exists():
            for file_path in core_dir.glob("*.py"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 简单分析模块
                    module_info = {
                        "name": file_path.stem,
                        "path": str(file_path.relative_to(self.project_root)),
                        "size": file_path.stat().st_size,
                        "functions": content.count('def '),
                        "classes": content.count('class '),
                        "imports": content.count('import'),
                        "description": self._extract_module_description(content)
                    }
                    
                    modules.append(module_info)
                    
                except Exception as e:
                    print(f"⚠️ 分析模块失败 {file_path}: {e}")
        
        return modules
    
    def _extract_module_description(self, content: str) -> str:
        """提取模块描述"""
        lines = content.split('\n')
        for line in lines[:10]:  # 只检查前10行
            if line.strip().startswith('"""') or line.strip().startswith("'''"):
                return "有文档字符串的模块"
            elif 'engine' in line.lower():
                return "引擎相关模块"
            elif 'cache' in line.lower():
                return "缓存相关模块"
            elif 'security' in line.lower():
                return "安全相关模块"
            elif 'workflow' in line.lower():
                return "工作流相关模块"
        
        return "通用功能模块"
    
    async def _get_dependencies(self) -> Dict[str, Any]:
        """获取依赖信息"""
        dependencies = {
            "python_version": "3.10+",
            "core_libraries": [],
            "external_libraries": [],
            "dev_dependencies": []
        }
        
        # 读取requirements文件
        req_files = ["requirements.txt", "requirements-dev.txt", "pyproject.toml"]
        
        for req_file in req_files:
            file_path = self.project_root / req_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if req_file == "pyproject.toml":
                        # 解析pyproject.toml
                        pass  # 简化处理
                    else:
                        # 解析requirements文件
                        lines = content.split('\n')
                        for line in lines:
                            if line.strip() and not line.startswith('#'):
                                if 'dev' in req_file:
                                    dependencies["dev_dependencies"].append(line.strip())
                                else:
                                    dependencies["external_libraries"].append(line.strip())
                
                except Exception as e:
                    print(f"⚠️ 读取依赖文件失败 {req_file}: {e}")
        
        return dependencies
    
    async def _get_configuration(self) -> Dict[str, Any]:
        """获取配置信息"""
        config = {
            "project_config": {},
            "build_config": {},
            "test_config": {}
        }
        
        # 读取配置文件
        config_files = ["pyproject.toml", "setup.cfg", "pytest.ini"]
        
        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    config["project_config"][config_file] = {
                        "exists": True,
                        "size": len(content),
                        "last_modified": file_path.stat().st_mtime
                    }
                
                except Exception as e:
                    print(f"⚠️ 读取配置文件失败 {config_file}: {e}")
        
        return config
    
    async def _get_recent_changes(self) -> List[Dict[str, Any]]:
        """获取最近变更"""
        changes = []
        
        # 获取最近修改的文件
        import time
        current_time = time.time()
        one_day_ago = current_time - 24 * 60 * 60
        
        for root, dirs, files in os.walk(self.project_root):
            # 跳过隐藏目录
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if file_path.stat().st_mtime > one_day_ago:
                        changes.append({
                            "file": str(file_path.relative_to(self.project_root)),
                            "modified_time": file_path.stat().st_mtime,
                            "size": file_path.stat().st_size
                        })
        
        return sorted(changes, key=lambda x: x["modified_time"], reverse=True)[:10]
    
    async def _get_critical_files(self) -> List[Dict[str, Any]]:
        """获取关键文件"""
        critical_patterns = [
            "*.py",
            "*.toml",
            "*.cfg",
            "*.ini",
            "*.md",
            "*.bat"
        ]
        
        critical_files = []
        
        for pattern in critical_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    critical_files.append({
                        "name": file_path.name,
                        "path": str(file_path.relative_to(self.project_root)),
                        "size": file_path.stat().st_size,
                        "type": file_path.suffix
                    })
        
        return critical_files[:50]  # 限制数量
    
    async def _save_contexts(self, contexts: Dict[str, Any]) -> Dict[str, str]:
        """保存上下文文件"""
        saved_files = {}
        
        for context_name, context_data in contexts.items():
            file_path = self.ai_context_dir / f"{context_name}.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(context_data, f, ensure_ascii=False, indent=2)
            
            saved_files[context_name] = str(file_path)
        
        return saved_files
    
    async def _create_force_commands(self, context_files: Dict[str, str]) -> List[str]:
        """创建强制传递指令"""
        commands = []
        
        commands.append("# AI强制信息传递指令")
        commands.append("## 必须执行的操作")
        commands.append("")
        commands.append("1. **立即读取所有上下文文件**:")
        for name, path in context_files.items():
            commands.append(f"   - `{path}`")
        
        commands.append("")
        commands.append("2. **完全理解项目状态**:")
        commands.append("   - 阅读项目上下文了解整体架构")
        commands.append("   - 理解测试要求明确分析目标")
        commands.append("   - 掌握决策框架确保分析质量")
        commands.append("   - 熟悉功能分类准确评估价值")
        
        commands.append("")
        commands.append("3. **严格执行分析标准**:")
        commands.append("   - 每个文件决策必须提供完整依据")
        commands.append("   - 所有推理过程必须详细记录")
        commands.append("   - 功能特点分析必须全面客观")
        commands.append("   - 保留/删除决策必须有充分证据")
        
        commands.append("")
        commands.append("4. **输出格式要求**:")
        commands.append("   - 使用结构化报告格式")
        commands.append("   - 包含完整的推理链条")
        commands.append("   - 提供具体的改进建议")
        commands.append("   - 标注所有数据来源")
        
        return commands
    
    async def _create_ai_verification(self) -> Dict[str, Any]:
        """创建AI理解验证"""
        verification = {
            "verification_questions": [
                "你是否完全理解项目的整体架构和核心功能？",
                "你是否清楚测试的具体要求和成功标准？",
                "你是否掌握了文件分析的决策框架？",
                "你是否了解功能分类和评估标准？"
            ],
            "understanding_check": {
                "project_name": "iFlow CLI V16 Quantum Evolution",
                "core_components": ["ARQ引擎", "HRRK内核", "REFRAG系统"],
                "test_objectives": ["测试覆盖", "代码质量", "安全扫描", "性能测试"],
                "decision_principles": ["证据驱动", "影响最小化", "价值最大化"]
            },
            "quality_assurance": [
                "确保所有分析都有具体数据支撑",
                "确保所有决策都有详细推理过程",
                "确保所有建议都有可行性评估",
                "确保所有结论都有验证方法"
            ]
        }
        
        return verification

# 使用示例
async def main():
    """主函数"""
    project_root = "."  # 当前目录
    
    forcer = AIInformationForcer(project_root)
    result = await forcer.force_ai_awareness()
    
    print("🎯 AI强制信息传递结果:")
    print(f"状态: {result['status']}")
    print(f"上下文文件: {len(result['context_files'])}个")
    print(f"强制指令: {len(result['force_commands'])}条")
    
    # 保存强制指令到文件
    commands_file = Path(project_root) / ".iflow" / "temp_docs" / "ai_force_commands.md"
    with open(commands_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result['force_commands']))
    
    print(f"📝 强制指令已保存到: {commands_file}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())