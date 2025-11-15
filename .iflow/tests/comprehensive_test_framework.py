#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 综合测试框架 V1.0
Comprehensive Testing Framework V1.0

你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import asyncio
import json
import logging
import time
import traceback
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import sys
import os

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from path_manager import get_path_manager
    from core.performance_optimizer import get_performance_optimizer
except ImportError as e:
    print(f"警告: 无法导入依赖模块: {e}")
    get_path_manager = None
    get_performance_optimizer = None

logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    status: str  # passed, failed, skipped, error
    duration: float
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TestSuite:
    """测试套件"""
    name: str
    tests: List[TestResult] = field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    total_duration: float = 0.0
    success_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class ComprehensiveTestFramework:
    """综合测试框架"""
    
    def __init__(self):
        """初始化测试框架"""
        self.path_manager = get_path_manager() if get_path_manager else None
        self.performance_optimizer = get_performance_optimizer() if get_performance_optimizer else None
        self.test_suites = []
        self.test_categories = {
            'unit': [],      # 单元测试
            'integration': [], # 集成测试
            'performance': [], # 性能测试
            'security': [],   # 安全测试
            'compatibility': [] # 兼容性测试
        }
        
        # 配置日志
        self._setup_logging()
        
        logger.info("🧪 综合测试框架初始化完成")
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = self.path_manager.log_dir if self.path_manager else Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 测试日志文件
        test_log_file = log_dir / f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 配置测试日志
        test_logger = logging.getLogger("test_framework")
        test_logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(test_log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        test_logger.addHandler(file_handler)
        test_logger.addHandler(console_handler)
        
        self.test_logger = test_logger
        self.test_log_file = test_log_file
    
    def register_test(self, category: str, test_func: Callable, test_name: Optional[str] = None):
        """注册测试"""
        if category not in self.test_categories:
            self.test_categories[category] = []
        
        test_info = {
            'func': test_func,
            'name': test_name or test_func.__name__,
            'category': category
        }
        
        self.test_categories[category].append(test_info)
        self.test_logger.info(f"📝 注册测试: {category}.{test_info['name']}")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        self.test_logger.info("🚀 开始运行所有测试...")
        
        all_results = {}
        overall_start = time.time()
        
        for category, tests in self.test_categories.items():
            if tests:
                self.test_logger.info(f"📂 运行 {category} 测试套件...")
                suite_result = await self._run_test_suite(category, tests)
                all_results[category] = asdict(suite_result)
                self.test_suites.append(suite_result)
        
        overall_duration = time.time() - overall_start
        
        # 计算总体统计
        total_tests = sum(suite.total_tests for suite in self.test_suites)
        total_passed = sum(suite.passed_tests for suite in self.test_suites)
        total_failed = sum(suite.failed_tests for suite in self.test_suites)
        total_skipped = sum(suite.skipped_tests for suite in self.test_suites)
        total_errors = sum(suite.error_tests for suite in self.test_suites)
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_duration': overall_duration,
            'total_tests': total_tests,
            'passed_tests': total_passed,
            'failed_tests': total_failed,
            'skipped_tests': total_skipped,
            'error_tests': total_errors,
            'overall_success_rate': overall_success_rate,
            'test_suites': all_results,
            'log_file': str(self.test_log_file)
        }
        
        self.test_logger.info(f"✅ 测试完成 - 总体成功率: {overall_success_rate:.1f}%")
        
        # 保存测试报告
        await self._save_test_report(summary)
        
        return summary
    
    async def _run_test_suite(self, category: str, tests: List[Dict[str, Any]]) -> TestSuite:
        """运行测试套件"""
        suite = TestSuite(name=category)
        suite_start = time.time()
        
        for test_info in tests:
            test_result = await self._run_single_test(test_info)
            suite.tests.append(test_result)
            
            # 更新统计
            suite.total_tests += 1
            suite.total_duration += test_result.duration
            
            if test_result.status == 'passed':
                suite.passed_tests += 1
            elif test_result.status == 'failed':
                suite.failed_tests += 1
            elif test_result.status == 'skipped':
                suite.skipped_tests += 1
            elif test_result.status == 'error':
                suite.error_tests += 1
        
        suite.success_rate = (suite.passed_tests / suite.total_tests * 100) if suite.total_tests > 0 else 0
        suite.duration = time.time() - suite_start
        
        self.test_logger.info(f"📊 {category} 套件完成 - 成功率: {suite.success_rate:.1f}%")
        
        return suite
    
    async def _run_single_test(self, test_info: Dict[str, Any]) -> TestResult:
        """运行单个测试"""
        test_func = test_info['func']
        test_name = test_info['name']
        category = test_info['category']
        
        self.test_logger.info(f"🔍 运行测试: {category}.{test_name}")
        
        start_time = time.time()
        
        try:
            # 如果有性能优化器，使用优化执行
            if self.performance_optimizer:
                result = await self.performance_optimizer.execute_with_optimization(test_func)
            else:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
            
            duration = time.time() - start_time
            
            # 检查结果
            if result is True or result is None:
                status = 'passed'
                error_message = None
                traceback_info = None
            else:
                status = 'failed'
                error_message = str(result)
                traceback_info = "测试返回False"
        
        except AssertionError as e:
            duration = time.time() - start_time
            status = 'failed'
            error_message = str(e)
            traceback_info = traceback.format_exc()
        
        except Exception as e:
            duration = time.time() - start_time
            status = 'error'
            error_message = str(e)
            traceback_info = traceback.format_exc()
        
        # 收集性能指标
        metrics = {}
        if self.performance_optimizer:
            metrics = self.performance_optimizer.get_performance_report()
        
        test_result = TestResult(
            test_name=test_name,
            status=status,
            duration=duration,
            error_message=error_message,
            traceback=traceback_info,
            metrics=metrics
        )
        
        status_icon = {"passed": "✅", "failed": "❌", "skipped": "⏭️", "error": "🚨"}[status]
        self.test_logger.info(f"{status_icon} {category}.{test_name}: {status} ({duration:.3f}s)")
        
        return test_result
    
    async def _save_test_report(self, summary: Dict[str, Any]):
        """保存测试报告"""
        if not self.path_manager:
            return
        
        reports_dir = self.path_manager.project_root / "test_reports"
        reports_dir.mkdir(exist_ok=True)
        
        # JSON报告
        json_file = reports_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
        
        # HTML报告
        html_file = reports_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        html_content = self._generate_html_report(summary)
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.test_logger.info(f"📄 测试报告已保存: {json_file}, {html_file}")
    
    def _generate_html_report(self, summary: Dict[str, Any]) -> str:
        """生成HTML报告"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>测试报告 - {summary['timestamp']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .test-suite {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .test-suite h3 {{ background: #f8f8f8; margin: 0; padding: 10px; }}
        .test-results {{ padding: 10px; }}
        .test-result {{ margin: 5px 0; padding: 5px; border-radius: 3px; }}
        .passed {{ background: #d4edda; }}
        .failed {{ background: #f8d7da; }}
        .error {{ background: #fff3cd; }}
        .skipped {{ background: #e2e3e5; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 综合测试报告</h1>
        <p><strong>时间:</strong> {summary['timestamp']}</p>
        <p><strong>总耗时:</strong> {summary['total_duration']:.2f}秒</p>
    </div>
    
    <div class="summary">
        <h2>📊 总体统计</h2>
        <table>
            <tr><th>项目</th><th>数值</th></tr>
            <tr><td>总测试数</td><td>{summary['total_tests']}</td></tr>
            <tr><td>通过</td><td>{summary['passed_tests']}</td></tr>
            <tr><td>失败</td><td>{summary['failed_tests']}</td></tr>
            <tr><td>跳过</td><td>{summary['skipped_tests']}</td></tr>
            <tr><td>错误</td><td>{summary['error_tests']}</td></tr>
            <tr><td>成功率</td><td>{summary['overall_success_rate']:.1f}%</td></tr>
        </table>
    </div>
"""
        
        # 添加各测试套件的详细结果
        for category, suite_data in summary['test_suites'].items():
            html += f"""
    <div class="test-suite">
        <h3>📂 {category.title()} 测试套件</h3>
        <div class="test-results">
            <p><strong>成功率:</strong> {suite_data['success_rate']:.1f}% ({suite_data['passed_tests']}/{suite_data['total_tests']})</p>
"""
            
            # 添加测试结果列表
            for test in suite_data.get('tests', []):
                css_class = test['status']
                html += f"""
            <div class="test-result {css_class}">
                <strong>{test['test_name']}</strong>: {test['status']} 
                ({test['duration']:.3f}s)
                {f'<br><em>错误: {test["error_message"]}</em>' if test.get('error_message') else ''}
            </div>
"""
            
            html += """
        </div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html

# 内置测试函数
async def test_path_manager():
    """测试路径管理器"""
    if not get_path_manager:
        raise ImportError("PathManager不可用")
    
    pm = get_path_manager()
    assert pm.project_root.exists(), "项目根目录不存在"
    assert pm.tools_dir.exists(), "工具目录不存在"
    assert len(pm.get_python_files()) > 0, "没有找到Python文件"
    
    return True

async def test_performance_optimizer():
    """测试性能优化器"""
    if not get_performance_optimizer:
        raise ImportError("PerformanceOptimizer不可用")
    
    optimizer = get_performance_optimizer()
    report = optimizer.get_performance_report()
    assert report is not None, "性能报告为空"
    
    return True

async def test_dependencies():
    """测试依赖"""
    dependencies = ['numpy', 'psutil', 'asyncio']
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            raise ImportError(f"依赖 {dep} 不可用")
    
    return True

async def test_file_structure():
    """测试文件结构"""
    if not get_path_manager:
        raise ImportError("PathManager不可用")
    
    pm = get_path_manager()
    
    # 检查核心目录
    required_dirs = ['core', 'tools', 'tests', 'hooks']
    for dir_name in required_dirs:
        dir_path = pm.project_root / ".iflow" / dir_name
        if not dir_path.exists():
            raise AssertionError(f"目录 {dir_name} 不存在")
    
    return True

async def main():
    """主函数 - 运行测试"""
    framework = ComprehensiveTestFramework()
    
    # 注册内置测试
    framework.register_test('unit', test_path_manager, "路径管理器测试")
    framework.register_test('unit', test_performance_optimizer, "性能优化器测试")
    framework.register_test('integration', test_dependencies, "依赖测试")
    framework.register_test('integration', test_file_structure, "文件结构测试")
    
    # 运行所有测试
    results = await framework.run_all_tests()
    
    print("\n🎉 测试完成!")
    print(f"总体成功率: {results['overall_success_rate']:.1f}%")
    print(f"详细报告: {results['log_file']}")

if __name__ == "__main__":
    asyncio.run(main())
