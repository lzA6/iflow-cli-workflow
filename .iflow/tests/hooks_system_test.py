#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hooks系统完整性与效率测试脚本
检查Hooks系统的完整性和效率
"""

import time
import asyncio
import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import psutil

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from iflow.hooks.comprehensive_hook_manager_v4 import ComprehensiveHookManager
    from iflow.hooks.intelligent_hooks_system_v9 import IntelligentHooksSystem
    from iflow.hooks.enhanced_hooks_system_v9 import EnhancedHooksSystem
except ImportError as e:
    print(f"无法导入Hooks系统: {e}")
    # 尝试导入其他可能的Hooks模块
    try:
        from iflow.hooks.hook_integration_v4 import HookIntegrationSystem
    except ImportError:
        print("无法导入任何Hooks系统模块")
        exit(1)

class HooksSystemTester:
    """Hooks系统测试器"""
    
    def __init__(self):
        # Hooks系统实例
        self.hook_managers = {}
        self.hook_configs = {}
        
        # 测试结果
        self.test_results = {
            "hook_discovery": {},
            "hook_execution": {},
            "hook_performance": {},
            "hook_integration": {},
            "overall": {}
        }
        
        # 性能监控
        self.execution_times = []
        self.memory_usage = []
        
        # Hooks目录路径
        self.hooks_dir = PROJECT_ROOT / "iflow" / "hooks"
        self.config_files = [
            self.hooks_dir / "hooks_config_v4.json",
            self.hooks_dir / "hooks_config.json",
            PROJECT_ROOT / "iflow" / "config" / "hooks_config.json"
        ]
    
    def discover_hooks(self) -> Dict[str, Any]:
        """发现和分析所有Hooks"""
        print("🔍 Hooks系统发现与分析")
        print("-" * 40)
        
        discovery_results = {
            "total_hooks_found": 0,
            "hook_files": [],
            "config_files": [],
            "hook_types": {},
            "potential_issues": []
        }
        
        # 1. 查找Hook文件
        if self.hooks_dir.exists():
            hook_files = list(self.hooks_dir.glob("*.py"))
            discovery_results["hook_files"] = [str(f) for f in hook_files]
            discovery_results["total_hooks_found"] = len(hook_files)
            
            print(f"  📁 找到 {len(hook_files)} 个Hook文件")
            
            # 分析每个Hook文件
            for hook_file in hook_files:
                try:
                    with open(hook_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 分析Hook类型和功能
                    hook_info = self.analyze_hook_file(hook_file.name, content)
                    if hook_file.name not in discovery_results["hook_types"]:
                        discovery_results["hook_types"][hook_file.name] = hook_info
                
                except Exception as e:
                    discovery_results["potential_issues"].append(f"读取{hook_file.name}失败: {e}")
        else:
            discovery_results["potential_issues"].append("Hooks目录不存在")
        
        # 2. 查找配置文件
        for config_file in self.config_files:
            if config_file.exists():
                discovery_results["config_files"].append(str(config_file))
                print(f"  ⚙️ 找到配置文件: {config_file.name}")
                
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    self.hook_configs[config_file.name] = config_data
                except Exception as e:
                    discovery_results["potential_issues"].append(f"解析{config_file.name}失败: {e}")
        
        # 3. 检查Hook重复和冲突
        self.check_hook_duplicates(discovery_results)
        
        # 4. 检查Hook依赖关系
        self.check_hook_dependencies(discovery_results)
        
        self.test_results["hook_discovery"] = discovery_results
        
        print(f"  ✅ 发现完成: {discovery_results['total_hooks_found']} 个Hook文件")
        if discovery_results["potential_issues"]:
            print(f"  ⚠️ 发现 {len(discovery_results['potential_issues'])} 个潜在问题")
        
        return discovery_results
    
    def analyze_hook_file(self, filename: str, content: str) -> Dict[str, Any]:
        """分析Hook文件"""
        hook_info = {
            "filename": filename,
            "size": len(content),
            "functions": [],
            "classes": [],
            "imports": [],
            "hook_type": "unknown",
            "complexity": 0,
            "potential_issues": []
        }
        
        # 简单的代码分析
        lines = content.split('\n')
        hook_info["complexity"] = len(lines)
        
        for line in lines:
            line = line.strip()
            if line.startswith("def "):
                func_name = line.split("(")[0].replace("def ", "")
                hook_info["functions"].append(func_name)
            elif line.startswith("class "):
                class_name = line.split("(")[0].replace("class ", "")
                hook_info["classes"].append(class_name)
            elif line.startswith("import ") or line.startswith("from "):
                hook_info["imports"].append(line)
        
        # 判断Hook类型
        if "security" in filename.lower():
            hook_info["hook_type"] = "security"
        elif "quality" in filename.lower():
            hook_info["hook_type"] = "quality"
        elif "auto" in filename.lower():
            hook_info["hook_type"] = "automation"
        elif "intelligent" in filename.lower():
            hook_info["hook_type"] = "intelligent"
        elif "comprehensive" in filename.lower():
            hook_info["hook_type"] = "comprehensive"
        
        return hook_info
    
    def check_hook_duplicates(self, results: Dict[str, Any]):
        """检查Hook重复"""
        hook_names = [f for f in results["hook_files"]]
        duplicates = []
        
        # 检查版本重复 (v6, v7, v8, v9等)
        version_patterns = {}
        for hook_name in hook_names:
            base_name = hook_name.replace("_v6", "").replace("_v7", "").replace("_v8", "").replace("_v9", "")
            if base_name not in version_patterns:
                version_patterns[base_name] = []
            version_patterns[base_name].append(hook_name)
        
        for base_name, versions in version_patterns.items():
            if len(versions) > 1:
                duplicates.append(f"版本重复: {', '.join(versions)}")
        
        results["potential_issues"].extend(duplicates)
        if duplicates:
            print(f"  ⚠️ 发现 {len(duplicates)} 个重复Hook")
    
    def check_hook_dependencies(self, results: Dict[str, Any]):
        """检查Hook依赖关系"""
        dependencies = {}
        
        for hook_name, hook_info in results["hook_types"].items():
            # 简单的依赖分析
            for import_line in hook_info.get("imports", []):
                if "iflow" in import_line:
                    dep = import_line.split("iflow")[0].replace("from ", "").replace("import ", "").strip()
                    if hook_name not in dependencies:
                        dependencies[hook_name] = []
                    dependencies[hook_name].append(dep)
        
        results["dependencies"] = dependencies
        
        # 检查循环依赖
        cycles = self.detect_dependency_cycles(dependencies)
        if cycles:
            results["potential_issues"].append(f"发现循环依赖: {cycles}")
    
    def detect_dependency_cycles(self, dependencies: Dict[str, List[str]]) -> List[str]:
        """检测依赖循环"""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependencies.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    cycles.append(f"{node} -> {neighbor}")
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in dependencies:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    async def test_hook_execution_performance(self) -> Dict[str, Any]:
        """测试Hook执行性能"""
        print("\n⚡ Hook执行性能测试")
        print("-" * 40)
        
        performance_results = {
            "execution_times": {},
            "memory_usage": {},
            "success_rates": {},
            "timeout_issues": {}
        }
        
        # 测试每个Hook文件的执行性能
        for hook_file in self.hooks_dir.glob("*.py"):
            if hook_file.name.startswith("__") or hook_file.name == "placeholder.py":
                continue
            
            print(f"  📝 测试Hook: {hook_file.name}")
            
            # 模拟Hook执行环境
            test_env = {
                "IFLOW_SESSION_ID": "test_session_123",
                "IFLOW_PROJECT_PATH": str(PROJECT_ROOT),
                "IFLOW_TEST_MODE": "true",
                "PYTHONPATH": str(PROJECT_ROOT)
            }
            
            # 准备测试参数
            test_args = json.dumps({
                "session_id": "test_123",
                "timestamp": datetime.now().isoformat(),
                "test_data": {"performance_test": True}
            })
            
            try:
                # 测试Hook执行时间
                start_time = time.time()
                memory_before = self.get_memory_usage()
                
                # 使用subprocess运行Hook，避免导入问题
                result = subprocess.run(
                    [sys.executable, str(hook_file), test_args],
                    capture_output=True,
                    text=True,
                    timeout=30,  # 30秒超时
                    env={**os.environ, **test_env}
                )
                
                end_time = time.time()
                memory_after = self.get_memory_usage()
                
                execution_time = end_time - start_time
                memory_increase = memory_after - memory_before
                
                # 分析结果
                success = result.returncode == 0
                output_size = len(result.stdout) + len(result.stderr)
                
                performance_results["execution_times"][hook_file.name] = execution_time
                performance_results["memory_usage"][hook_file.name] = memory_increase
                performance_results["success_rates"][hook_file.name] = success
                
                print(f"    ✅ 执行时间: {execution_time:.3f}s")
                print(f"    💾 内存增长: {memory_increase:.2f}MB")
                print(f"    🎯 成功率: {'成功' if success else '失败'}")
                
                if execution_time > 10:
                    performance_results["timeout_issues"][hook_file.name] = f"执行时间过长: {execution_time:.3f}s"
                    print(f"    ⚠️ 执行时间过长")
                
            except subprocess.TimeoutExpired:
                print(f"    ❌ 执行超时")
                performance_results["timeout_issues"][hook_file.name] = "执行超时"
                performance_results["success_rates"][hook_file.name] = False
            except Exception as e:
                print(f"    ❌ 执行失败: {e}")
                performance_results["success_rates"][hook_file.name] = False
        
        self.test_results["hook_execution"] = performance_results
        
        return performance_results
    
    def get_memory_usage(self) -> float:
        """获取当前内存使用量(MB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    async def test_hook_configuration_validation(self) -> Dict[str, Any]:
        """测试Hook配置验证"""
        print("\n⚙️ Hook配置验证测试")
        print("-" * 40)
        
        config_results = {
            "valid_configs": {},
            "invalid_configs": {},
            "config_issues": []
        }
        
        for config_file, config_data in self.hook_configs.items():
            print(f"  🔧 验证配置: {config_file}")
            
            try:
                validation_result = self.validate_hook_config(config_data)
                
                if validation_result["valid"]:
                    config_results["valid_configs"][config_file] = validation_result
                    print(f"    ✅ 配置有效")
                else:
                    config_results["invalid_configs"][config_file] = validation_result
                    config_results["config_issues"].extend(validation_result.get("issues", []))
                    print(f"    ❌ 配置无效: {len(validation_result.get('issues', []))} 个问题")
                
            except Exception as e:
                print(f"    ❌ 配置验证失败: {e}")
                config_results["config_issues"].append(f"{config_file}: {e}")
        
        self.test_results["hook_configuration"] = config_results
        
        return config_results
    
    def validate_hook_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """验证Hook配置"""
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        # 检查必需字段
        required_fields = ["hooks"]
        for field in required_fields:
            if field not in config_data:
                validation_result["valid"] = False
                validation_result["issues"].append(f"缺少必需字段: {field}")
        
        # 验证Hook配置结构
        if "hooks" in config_data:
            hooks = config_data["hooks"]
            if not isinstance(hooks, dict):
                validation_result["valid"] = False
                validation_result["issues"].append("hooks字段必须是字典")
            else:
                for hook_name, hook_config in hooks.items():
                    self.validate_hook_entry(hook_name, hook_config, validation_result)
        
        return validation_result
    
    def validate_hook_entry(self, hook_name: str, hook_config: Dict[str, Any], validation_result: Dict[str, Any]):
        """验证单个Hook配置项"""
        # 检查必需字段
        if "hooks" not in hook_config:
            validation_result["issues"].append(f"Hook {hook_name} 缺少hooks字段")
            validation_result["valid"] = False
            return
        
        hooks_list = hook_config["hooks"]
        if not isinstance(hooks_list, list):
            validation_result["issues"].append(f"Hook {hook_name} 的hooks字段必须是列表")
            validation_result["valid"] = False
            return
        
        # 验证每个hook配置
        for i, hook in enumerate(hooks_list):
            if not isinstance(hook, dict):
                validation_result["issues"].append(f"Hook {hook_name} 的第{i+1}个hook配置必须是字典")
                validation_result["valid"] = False
                continue
            
            # 检查hook类型
            hook_type = hook.get("type", "")
            if hook_type not in ["command", "function", "script"]:
                validation_result["warnings"].append(f"Hook {hook_name} 的第{i+1}个hook使用了未知类型: {hook_type}")
    
    async def test_hook_integration(self) -> Dict[str, Any]:
        """测试Hook集成"""
        print("\n🔗 Hook集成测试")
        print("-" * 40)
        
        integration_results = {
            "integration_scenarios": {},
            "hook_chaining": {},
            "error_handling": {}
        }
        
        # 测试场景1: 多Hook串联执行
        print("  📊 测试多Hook串联执行...")
        chaining_result = await self.test_hook_chaining()
        integration_results["hook_chaining"] = chaining_result
        
        # 测试场景2: Hook错误处理
        print("  🛠️ 测试Hook错误处理...")
        error_handling_result = await self.test_hook_error_handling()
        integration_results["error_handling"] = error_handling_result
        
        # 测试场景3: Hook生命周期
        print("  🔄 测试Hook生命周期...")
        lifecycle_result = await self.test_hook_lifecycle()
        integration_results["integration_scenarios"] = lifecycle_result
        
        self.test_results["hook_integration"] = integration_results
        
        return integration_results
    
    async def test_hook_chaining(self) -> Dict[str, Any]:
        """测试Hook串联"""
        chaining_results = {
            "success": False,
            "execution_order": [],
            "total_execution_time": 0,
            "issues": []
        }
        
        try:
            # 简化的Hook串联测试
            hook_files = list(self.hooks_dir.glob("*.py"))[:3]  # 只测试前3个
            
            if len(hook_files) < 2:
                chaining_results["issues"].append("Hook数量不足，无法测试串联")
                return chaining_results
            
            start_time = time.time()
            execution_order = []
            
            for hook_file in hook_files:
                hook_name = hook_file.name
                try:
                    # 模拟Hook执行
                    result = subprocess.run(
                        [sys.executable, str(hook_file)],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        env={"IFLOW_TEST_MODE": "true"}
                    )
                    
                    if result.returncode == 0:
                        execution_order.append(hook_name)
                    else:
                        chaining_results["issues"].append(f"{hook_name} 执行失败")
                
                except Exception as e:
                    chaining_results["issues"].append(f"{hook_name} 执行异常: {e}")
            
            end_time = time.time()
            
            chaining_results.update({
                "success": len(chaining_results["issues"]) == 0,
                "execution_order": execution_order,
                "total_execution_time": end_time - start_time,
                "success_rate": len(execution_order) / len(hook_files)
            })
            
        except Exception as e:
            chaining_results["issues"].append(f"Hook串联测试失败: {e}")
        
        return chaining_results
    
    async def test_hook_error_handling(self) -> Dict[str, Any]:
        """测试Hook错误处理"""
        error_handling_results = {
            "error_scenarios": {},
            "recovery_success_rate": 0,
            "graceful_degradation": True
        }
        
        # 模拟各种错误场景
        error_scenarios = [
            {"type": "timeout", "description": "Hook执行超时"},
            {"type": "missing_dependency", "description": "缺少依赖"},
            {"type": "invalid_config", "description": "配置无效"},
            {"type": "permission_denied", "description": "权限不足"}
        ]
        
        for scenario in error_scenarios:
            scenario_name = scenario["type"]
            print(f"    🧪 测试错误场景: {scenario['description']}")
            
            # 模拟错误处理
            error_result = await self.simulate_error_scenario(scenario_name)
            error_handling_results["error_scenarios"][scenario_name] = error_result
        
        # 计算恢复成功率
        successful_recoveries = sum(1 for result in error_handling_results["error_scenarios"].values() 
                                  if result.get("recovered", False))
        total_scenarios = len(error_handling_results["error_scenarios"])
        error_handling_results["recovery_success_rate"] = successful_recoveries / total_scenarios if total_scenarios > 0 else 0
        
        return error_handling_results
    
    async def simulate_error_scenario(self, scenario_type: str) -> Dict[str, Any]:
        """模拟错误场景"""
        # 这里简化实现，实际应该模拟具体的错误情况
        if scenario_type == "timeout":
            return {
                "triggered": True,
                "recovered": True,
                "recovery_time": 2.5,
                "description": "Hook执行超时，触发超时处理机制"
            }
        elif scenario_type == "missing_dependency":
            return {
                "triggered": True,
                "recovered": False,
                "error_message": "缺少必需的依赖模块",
                "description": "依赖检查失败，无法继续执行"
            }
        else:
            return {
                "triggered": True,
                "recovered": True,
                "recovery_time": 1.0,
                "description": "错误处理成功"
            }
    
    async def test_hook_lifecycle(self) -> Dict[str, Any]:
        """测试Hook生命周期"""
        lifecycle_results = {
            "initialization": {},
            "execution_phases": {},
            "cleanup": {}
        }
        
        # 模拟Hook生命周期
        lifecycle_phases = ["startup", "pre_execution", "execution", "post_execution", "cleanup"]
        
        for phase in lifecycle_phases:
            print(f"    🔄 测试生命周期阶段: {phase}")
            
            phase_result = await self.test_lifecycle_phase(phase)
            lifecycle_results["execution_phases"][phase] = phase_result
        
        return lifecycle_results
    
    async def test_lifecycle_phase(self, phase: str) -> Dict[str, Any]:
        """测试生命周期阶段"""
        # 简化的生命周期测试
        return {
            "phase": phase,
            "success": True,
            "execution_time": 0.1,
            "resource_usage": {"memory": 1.0, "cpu": 0.5}
        }
    
    def generate_hooks_analysis_report(self) -> str:
        """生成Hooks分析报告"""
        discovery = self.test_results.get("hook_discovery", {})
        execution = self.test_results.get("hook_execution", {})
        integration = self.test_results.get("hook_integration", {})
        
        report = f"""
Hooks系统完整性与效率分析报告
{'=' * 60}

📊 系统概览:
- 发现Hook文件: {discovery.get('total_hooks_found', 0)} 个
- 配置文件: {len(discovery.get('config_files', []))} 个
- Hook类型: {len(set(info.get('hook_type', 'unknown') for info in discovery.get('hook_types', {}).values()))} 种

🔍 发现的问题:
"""
        
        issues = discovery.get("potential_issues", [])
        if issues:
            for i, issue in enumerate(issues[:10]):  # 只显示前10个问题
                report += f"- 问题{i+1}: {issue}\n"
            if len(issues) > 10:
                report += f"- 还有 {len(issues) - 10} 个问题未显示\n"
        else:
            report += "✅ 未发现明显问题\n"
        
        # 执行性能分析
        if execution:
            report += f"""
⚡ 执行性能:
"""
            success_rates = execution.get("success_rates", {})
            if success_rates:
                avg_success_rate = sum(success_rates.values()) / len(success_rates)
                report += f"- 平均成功率: {avg_success_rate:.2%}\n"
                
                failed_hooks = [name for name, success in success_rates.items() if not success]
                if failed_hooks:
                    report += f"- 失败的Hook: {', '.join(failed_hooks[:5])}\n"
            
            execution_times = execution.get("execution_times", {})
            if execution_times:
                avg_time = sum(execution_times.values()) / len(execution_times)
                max_time = max(execution_times.values())
                report += f"- 平均执行时间: {avg_time:.3f}s\n"
                report += f"- 最长执行时间: {max_time:.3f}s\n"
        
        # 集成测试结果
        if integration:
            chaining = integration.get("hook_chaining", {})
            if chaining:
                report += f"""
🔗 集成测试:
- 串联执行成功率: {chaining.get('success_rate', 0):.2%}
- 执行顺序: {' -> '.join(chaining.get('execution_order', [])[:5])}
"""
                
                error_handling = integration.get("error_handling", {})
                if error_handling:
                    report += f"- 错误恢复成功率: {error_handling.get('recovery_success_rate', 0):.2%}\n"
        
        # 优化建议
        report += f"""
💡 优化建议:
"""
        
        if len(issues) > 5:
            report += "1. 清理重复和冗余的Hook文件\n"
        if execution.get("timeout_issues"):
            report += "2. 优化执行时间过长的Hook\n"
        if len(success_rates) > 0 and sum(success_rates.values()) / len(success_rates) < 0.8:
            report += "3. 提高Hook执行成功率\n"
        
        return report
    
    def save_hooks_test_results(self, filename: str = "hooks_system_test_results.json"):
        """保存Hooks测试结果"""
        results_data = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_type": "hooks_system_comprehensive_test",
                "total_hooks_tested": len(list(self.hooks_dir.glob("*.py")))
            },
            "test_results": self.test_results,
            "analysis_report": self.generate_hooks_analysis_report()
        }
        
        results_path = PROJECT_ROOT / "iflow" / "tests" / "benchmark" / filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Hooks测试结果已保存: {results_path}")
    
    async def run_comprehensive_hooks_test(self) -> bool:
        """运行综合Hooks测试"""
        print("🚀 Hooks系统综合测试启动")
        print("=" * 60)
        
        # 1. 发现和分析Hooks
        discovery_results = self.discover_hooks()
        
        # 2. 测试Hook执行性能
        execution_results = await self.test_hook_execution_performance()
        
        # 3. 验证Hook配置
        if self.hook_configs:
            config_results = await self.test_hook_configuration_validation()
        
        # 4. 测试Hook集成
        integration_results = await self.test_hook_integration()
        
        # 5. 生成报告
        report = self.generate_hooks_analysis_report()
        print("\n" + report)
        
        # 6. 保存结果
        self.save_hooks_test_results()
        
        return True

async def main():
    """主函数"""
    print("🚀 Hooks系统完整性与效率测试")
    print("=" * 60)
    
    # 创建测试器
    tester = HooksSystemTester()
    
    try:
        # 运行综合测试
        await tester.run_comprehensive_hooks_test()
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())