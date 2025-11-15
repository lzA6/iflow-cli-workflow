#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能体指令验证器
用于验证所有智能体是否能通过/指令正确显示
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class AgentInstructionValidator:
    def __init__(self, agents_dir: str):
        self.agents_dir = Path(agents_dir)
        self.validation_results = {
            'total_agents': 0,
            'valid_agents': 0,
            'invalid_agents': 0,
            'errors': []
        }
    
    def validate_all_agents(self) -> Dict:
        """验证所有智能体的指令显示能力"""
        print("Starting agent instruction validation...")
        
        # 获取所有智能体目录
        agent_dirs = self._find_agent_directories()
        self.validation_results['total_agents'] = len(agent_dirs)
        
        for agent_dir in agent_dirs:
            agent_name = agent_dir.name
            is_valid, error = self._validate_agent(agent_dir)
            
            if is_valid:
                self.validation_results['valid_agents'] += 1
                print(f"✓ {agent_name}: VALID")
            else:
                self.validation_results['invalid_agents'] += 1
                self.validation_results['errors'].append({
                    'agent': agent_name,
                    'directory': str(agent_dir),
                    'error': error
                })
                print(f"✗ {agent_name}: INVALID - {error}")
        
        return self.validation_results
    
    def _find_agent_directories(self) -> List[Path]:
        """查找所有智能体目录"""
        agent_dirs = []
        
        # 检查agents目录下的子目录
        for item in self.agents_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # 检查是否包含README.md或智能体定义文件
                if self._has_agent_files(item):
                    agent_dirs.append(item)
        
        return agent_dirs
    
    def _has_agent_files(self, directory: Path) -> bool:
        """检查目录是否包含智能体文件"""
        # 检查README.md
        readme_file = directory / "README.md"
        if readme_file.exists():
            return True
        
        # 检查其他可能的智能体定义文件
        possible_files = ["agent.md", "definition.yaml", "config.json"]
        for filename in possible_files:
            if (directory / filename).exists():
                return True
        
        return False
    
    def _validate_agent(self, agent_dir: Path) -> Tuple[bool, str]:
        """验证单个智能体"""
        try:
            # 检查README.md是否存在
            readme_file = agent_dir / "README.md"
            if not readme_file.exists():
                return False, "Missing README.md file"
            
            # 验证README.md内容
            readme_content = readme_file.read_text(encoding='utf-8')
            if len(readme_content.strip()) < 10:
                return False, "README.md content too short"
            
            # 检查是否包含必要的元数据
            metadata = self._extract_metadata(readme_content)
            if not metadata:
                return False, "Missing or invalid metadata in README.md"
            
            # 验证元数据完整性
            required_fields = ['name', 'description', 'version']
            missing_fields = [field for field in required_fields if field not in metadata]
            if missing_fields:
                return False, f"Missing required metadata fields: {', '.join(missing_fields)}"
            
            # 检查指令兼容性
            instruction_compatibility = self._check_instruction_compatibility(metadata)
            if not instruction_compatibility:
                return False, "Instruction compatibility issues detected"
            
            return True, "Agent validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _extract_metadata(self, readme_content: str) -> Optional[Dict]:
        """从README.md提取元数据"""
        metadata = {}
        
        # 尝试提取YAML格式的元数据（如果在文件开头）
        if readme_content.startswith('---'):
            try:
                parts = readme_content.split('---')
                if len(parts) >= 3:
                    yaml_content = parts[1].strip()
                    metadata = yaml.safe_load(yaml_content) or {}
            except yaml.YAMLError:
                pass
        
        # 如果没有YAML元数据，尝试从标题提取基本信息
        if not metadata:
            lines = readme_content.split('\n')
            for line in lines[:10]:  # 检查前10行
                line = line.strip()
                if line.startswith('# ') and 'name' not in metadata:
                    metadata['name'] = line[2:].strip()
                elif line.startswith('> ') and 'description' not in metadata:
                    metadata['description'] = line[2:].strip()
        
        return metadata if metadata else None
    
    def _check_instruction_compatibility(self, metadata: Dict) -> bool:
        """检查指令兼容性"""
        # 检查是否有指令别名
        if 'commands' in metadata:
            commands = metadata['commands']
            if isinstance(commands, list) and len(commands) > 0:
                return True
        
        # 检查名称是否适合作为指令
        name = metadata.get('name', '')
        if name and len(name) <= 50 and name.replace('-', '').replace('_', '').isalnum():
            return True
        
        return False
    
    def generate_validation_report(self) -> str:
        """生成验证报告"""
        report = []
        report.append("=" * 70)
        report.append("Agent Instruction Validation Report")
        report.append("=" * 70)
        report.append(f"Total Agents Found: {self.validation_results['total_agents']}")
        report.append(f"Valid Agents: {self.validation_results['valid_agents']}")
        report.append(f"Invalid Agents: {self.validation_results['invalid_agents']}")
        report.append(f"Validation Rate: {self.validation_results['valid_agents']/max(self.validation_results['total_agents'], 1)*100:.1f}%")
        report.append("")
        
        if self.validation_results['errors']:
            report.append("Invalid Agents Details:")
            report.append("-" * 50)
            for error in self.validation_results['errors']:
                report.append(f"Agent: {error['agent']}")
                report.append(f"Directory: {error['directory']}")
                report.append(f"Error: {error['error']}")
                report.append("-" * 50)
        
        report.append("")
        report.append("Recommendations:")
        report.append("1. Ensure all agents have README.md files")
        report.append("2. Include proper metadata in README.md")
        report.append("3. Use YAML format for structured metadata")
        report.append("4. Define command aliases for agent accessibility")
        report.append("5. Follow naming conventions for agent directories")
        
        return "\n".join(report)
    
    def get_fix_suggestions(self, agent_dir: Path) -> List[str]:
        """获取修复建议"""
        suggestions = []
        
        # 检查README.md
        readme_file = agent_dir / "README.md"
        if not readme_file.exists():
            suggestions.append(f"Create README.md in {agent_dir.name}")
        else:
            # 检查元数据
            content = readme_file.read_text(encoding='utf-8')
            metadata = self._extract_metadata(content)
            
            if not metadata:
                suggestions.append(f"Add YAML metadata header to {agent_dir}/README.md")
            else:
                if 'name' not in metadata:
                    suggestions.append(f"Add 'name' field to metadata in {agent_dir}/README.md")
                if 'description' not in metadata:
                    suggestions.append(f"Add 'description' field to metadata in {agent_dir}/README.md")
                if 'version' not in metadata:
                    suggestions.append(f"Add 'version' field to metadata in {agent_dir}/README.md")
                if 'commands' not in metadata:
                    suggestions.append(f"Add 'commands' field to metadata in {agent_dir}/README.md")
        
        return suggestions

def main():
    """主函数"""
    # 获取agents目录
    current_file = Path(__file__)
    agents_dir = current_file.parent
    
    print("Agent Instruction Validator")
    print("=" * 50)
    print(f"Validating agents in: {agents_dir}")
    print("-" * 50)
    
    # 创建验证器
    validator = AgentInstructionValidator(agents_dir)
    
    # 执行验证
    results = validator.validate_all_agents()
    
    # 生成并显示报告
    report = validator.generate_validation_report()
    print("\n" + report)
    
    # 显示修复建议
    if results['invalid_agents'] > 0:
        print("\nFix Suggestions:")
        print("-" * 30)
        for error in results['errors']:
            agent_dir = Path(error['directory'])
            suggestions = validator.get_fix_suggestions(agent_dir)
            if suggestions:
                print(f"\n{error['agent']}:")
                for suggestion in suggestions:
                    print(f"  - {suggestion}")
    
    return results

if __name__ == "__main__":
    results = main()
    
    # 返回退出码
    exit(0 if results['invalid_agents'] == 0 else 1)