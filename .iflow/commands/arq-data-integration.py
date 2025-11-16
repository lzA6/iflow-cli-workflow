#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔗 ARQ数据集成命令 - 自动数据管理集成工具
==============================================

这个命令将ARQ数据管理功能无缝集成到工作流中，确保：
- 🔄 自动读取和调用本地数据集
- 📊 自动记录和分析会话数据
- 🧠 智能总结和查看历史会话
- 💾 自动同步知识库和偏好数据
- 🎯 无需手动命令的全自动运行

使用方法:
    python arq-data-integration.py [--auto-start] [--config config.json]

特性:
- 零配置自动启动
- 智能数据发现和集成
- 实时监控和优化
- 完整的错误处理和恢复

作者: AI架构师团队
版本: 17.0.0 Hyperdimensional Singularity
日期: 2025-11-17
"""

import os
import sys
import json
import asyncio
import logging
import argparse
import signal
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# 项目根路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / ".iflow" / "core"))

# 导入ARQ组件
try:
    from arq_data_manager_v17 import get_arq_data_manager, DataType, DataPriority
    from arq_data_analyzer_v17 import get_arq_data_analyzer
    ARQ_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ARQ组件不可用: {e}")
    ARQ_COMPONENTS_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ARQDataIntegration:
    """ARQ数据集成主类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化数据集成"""
        self.config = config or {}
        self.running = False
        
        # ARQ组件
        self.data_manager = None
        self.data_analyzer = None
        
        # 集成状态
        self.integration_status = {
            "started_at": None,
            "last_sync": None,
            "data_processed": 0,
            "errors_count": 0,
            "auto_operations": []
        }
        
        # 监控任务
        self.monitoring_tasks = []
        
        logger.info("🔗 ARQ数据集成初始化完成")
    
    async def initialize(self):
        """初始化ARQ组件"""
        if not ARQ_COMPONENTS_AVAILABLE:
            raise RuntimeError("ARQ组件不可用，无法初始化")
        
        logger.info("🚀 正在初始化ARQ组件...")
        
        # 初始化数据管理器
        self.data_manager = get_arq_data_manager()
        await self.data_manager.start_auto_sync()
        
        # 初始化数据分析器
        self.data_analyzer = get_arq_data_analyzer()
        
        # 创建默认项目会话
        await self._create_default_session()
        
        # 加载历史数据
        await self._load_historical_data()
        
        logger.info("✅ ARQ组件初始化完成")
    
    async def start_auto_integration(self):
        """启动自动集成"""
        if self.running:
            logger.warning("⚠️ 自动集成已在运行")
            return
        
        self.running = True
        self.integration_status["started_at"] = datetime.now()
        
        logger.info("🔄 启动ARQ自动数据集成...")
        
        # 启动监控任务
        await self._start_monitoring_tasks()
        
        # 执行初始数据同步
        await self._perform_initial_sync()
        
        # 启动主循环
        await self._main_integration_loop()
    
    async def stop_integration(self):
        """停止集成"""
        logger.info("⏹️ 正在停止ARQ数据集成...")
        
        self.running = False
        
        # 取消监控任务
        for task in self.monitoring_tasks:
            task.cancel()
        
        # 清理资源
        if self.data_manager:
            await self.data_manager.cleanup()
        
        # 保存集成状态
        await self._save_integration_status()
        
        logger.info("✅ ARQ数据集成已停止")
    
    async def _create_default_session(self):
        """创建默认会话"""
        try:
            session_id = await self.data_manager.create_session(
                project_id="arq_auto_integration",
                user_id="system",
                goals=["自动数据集成", "智能分析", "持续优化"]
            )
            
            # 存储会话ID到配置
            self.config["default_session_id"] = session_id
            
            logger.info(f"✅ 创建默认会话: {session_id}")
            
        except Exception as e:
            logger.error(f"❌ 创建默认会话失败: {e}")
    
    async def _load_historical_data(self):
        """加载历史数据"""
        try:
            # 扫描数据目录
            data_dir = PROJECT_ROOT / "data"
            if not data_dir.exists():
                logger.info("📁 数据目录不存在，跳过历史数据加载")
                return
            
            # 加载知识库数据
            await self._load_knowledge_base(data_dir)
            
            # 加载查询历史
            await self._load_query_history(data_dir)
            
            # 加载会话数据
            await self._load_session_data(data_dir)
            
            logger.info("✅ 历史数据加载完成")
            
        except Exception as e:
            logger.error(f"❌ 加载历史数据失败: {e}")
    
    async def _load_knowledge_base(self, data_dir: Path):
        """加载知识库数据"""
        kb_file = data_dir / "knowledge_base.json"
        if kb_file.exists():
            try:
                with open(kb_file, 'r', encoding='utf-8') as f:
                    kb_data = json.load(f)
                
                # 存储知识库数据
                for kb_id, kb_item in kb_data.items():
                    await self.data_manager.store_data(
                        data=kb_item,
                        data_type=DataType.KNOWLEDGE_BASE,
                        priority=DataPriority.HIGH,
                        tags={"knowledge_base", "auto_imported"}
                    )
                
                logger.info(f"📚 加载知识库数据: {len(kb_data)} 项")
                
            except Exception as e:
                logger.error(f"❌ 加载知识库失败: {e}")
    
    async def _load_query_history(self, data_dir: Path):
        """加载查询历史"""
        query_file = data_dir / "query_history.json"
        if query_file.exists():
            try:
                with open(query_file, 'r', encoding='utf-8') as f:
                    query_data = json.load(f)
                
                # 存储查询历史
                for query_item in query_data:
                    session_id = self.config.get("default_session_id")
                    if session_id:
                        await self.data_manager.record_query(
                            session_id=session_id,
                            query=query_item.get("question", ""),
                            context=query_item.get("context", ""),
                            response=query_item.get("response", {}).get("answer", ""),
                            response_time=query_item.get("response_time", 0.0),
                            confidence=query_item.get("response", {}).get("confidence", 0.0)
                        )
                
                logger.info(f"📝 加载查询历史: {len(query_data)} 条")
                
            except Exception as e:
                logger.error(f"❌ 加载查询历史失败: {e}")
    
    async def _load_session_data(self, data_dir: Path):
        """加载会话数据"""
        sessions_dir = data_dir / "sessions"
        if sessions_dir.exists():
            try:
                session_files = list(sessions_dir.glob("*.json"))
                
                for session_file in session_files:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    # 存储会话数据
                    await self.data_manager.store_data(
                        data=session_data,
                        data_type=DataType.SESSION_DATA,
                        tags={"session", "auto_imported"}
                    )
                
                logger.info(f"🔄 加载会话数据: {len(session_files)} 个文件")
                
            except Exception as e:
                logger.error(f"❌ 加载会话数据失败: {e}")
    
    async def _start_monitoring_tasks(self):
        """启动监控任务"""
        # 数据同步监控
        sync_task = asyncio.create_task(self._monitor_data_sync())
        self.monitoring_tasks.append(sync_task)
        
        # 性能监控
        perf_task = asyncio.create_task(self._monitor_performance())
        self.monitoring_tasks.append(perf_task)
        
        # 数据质量监控
        quality_task = asyncio.create_task(self._monitor_data_quality())
        self.monitoring_tasks.append(quality_task)
        
        logger.info("📊 监控任务已启动")
    
    async def _monitor_data_sync(self):
        """监控数据同步"""
        while self.running:
            try:
                # 检查同步状态
                summary = await self.data_manager.get_performance_summary()
                
                # 记录同步指标
                sync_info = {
                    "timestamp": datetime.now().isoformat(),
                    "cache_hit_rate": summary.get("cache_hit_rate", 0),
                    "active_sessions": summary.get("active_sessions", 0),
                    "memory_usage": summary.get("memory_usage", 0)
                }
                
                await self.data_manager.store_data(
                    data=sync_info,
                    data_type=DataType.SYSTEM_METRICS,
                    tags={"monitoring", "sync_status"}
                )
                
                await asyncio.sleep(300)  # 5分钟检查一次
                
            except Exception as e:
                logger.error(f"❌ 数据同步监控失败: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_performance(self):
        """监控性能"""
        while self.running:
            try:
                # 获取实时洞察
                insights = await self.data_analyzer.get_real_time_insights()
                
                # 检查性能问题
                if insights["system_status"] != "healthy":
                    logger.warning(f"⚠️ 系统性能警告: {insights['system_status']}")
                    
                    # 记录性能问题
                    await self.data_manager.store_data(
                        data=insights,
                        data_type=DataType.SYSTEM_METRICS,
                        priority=DataPriority.HIGH,
                        tags={"performance", "warning"}
                    )
                
                await asyncio.sleep(180)  # 3分钟检查一次
                
            except Exception as e:
                logger.error(f"❌ 性能监控失败: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_data_quality(self):
        """监控数据质量"""
        while self.running:
            try:
                # 每小时执行一次数据质量检查
                await self._perform_quality_check()
                
                await asyncio.sleep(3600)  # 1小时检查一次
                
            except Exception as e:
                logger.error(f"❌ 数据质量监控失败: {e}")
                await asyncio.sleep(300)
    
    async def _perform_quality_check(self):
        """执行数据质量检查"""
        try:
            # 分析使用模式
            usage_analysis = await self.data_analyzer.analyze_usage_patterns(
                time_range=timedelta(hours=1)
            )
            
            # 检查数据质量
            quality_score = usage_analysis.confidence
            
            if quality_score < 0.8:
                logger.warning(f"⚠️ 数据质量下降: {quality_score:.1%}")
                
                # 记录质量问题
                await self.data_manager.store_data(
                    data={
                        "quality_score": quality_score,
                        "analysis": asdict(usage_analysis),
                        "timestamp": datetime.now().isoformat()
                    },
                    data_type=DataType.SYSTEM_METRICS,
                    tags={"quality", "warning"}
                )
            
        except Exception as e:
            logger.error(f"❌ 数据质量检查失败: {e}")
    
    async def _perform_initial_sync(self):
        """执行初始同步"""
        try:
            logger.info("🔄 执行初始数据同步...")
            
            # 同步所有缓存数据
            await self.data_manager._sync_memory_to_db()
            
            # 执行数据库优化
            await self.data_manager._optimize_database()
            
            # 生成初始报告
            await self._generate_integration_report()
            
            self.integration_status["last_sync"] = datetime.now()
            
            logger.info("✅ 初始数据同步完成")
            
        except Exception as e:
            logger.error(f"❌ 初始同步失败: {e}")
    
    async def _main_integration_loop(self):
        """主集成循环"""
        logger.info("🔄 进入主集成循环...")
        
        while self.running:
            try:
                # 自动数据收集
                await self._auto_collect_data()
                
                # 智能数据分析
                await self._auto_analyze_data()
                
                # 自动优化
                await self._auto_optimize()
                
                # 更新集成状态
                self.integration_status["last_sync"] = datetime.now()
                
                # 等待下一个循环
                await asyncio.sleep(self.config.get("integration_interval", 600))  # 默认10分钟
                
            except Exception as e:
                logger.error(f"❌ 主集成循环错误: {e}")
                self.integration_status["errors_count"] += 1
                await asyncio.sleep(60)
    
    async def _auto_collect_data(self):
        """自动收集数据"""
        try:
            # 收集系统指标
            system_metrics = await self._collect_system_metrics()
            
            # 收集用户行为数据
            behavior_data = await self._collect_behavior_data()
            
            # 收集性能数据
            performance_data = await self._collect_performance_data()
            
            # 存储收集的数据
            for data_type, data in [
                (DataType.SYSTEM_METRICS, system_metrics),
                (DataType.USER_PREFERENCES, behavior_data),
                (DataType.SYSTEM_METRICS, performance_data)
            ]:
                if data:
                    await self.data_manager.store_data(
                        data=data,
                        data_type=data_type,
                        tags={"auto_collected"}
                    )
            
            self.integration_status["data_processed"] += 1
            
        except Exception as e:
            logger.error(f"❌ 自动数据收集失败: {e}")
    
    async def _auto_analyze_data(self):
        """自动分析数据"""
        try:
            # 定期执行分析
            current_time = datetime.now()
            
            # 检查是否需要执行分析
            last_analysis = self.config.get("last_analysis")
            if not last_analysis or (current_time - datetime.fromisoformat(last_analysis)).hours >= 1:
                
                # 执行使用模式分析
                await self.data_analyzer.analyze_usage_patterns(
                    time_range=timedelta(hours=24)
                )
                
                # 执行性能分析
                await self.data_analyzer.analyze_performance_metrics(
                    time_range=timedelta(hours=24)
                )
                
                # 更新最后分析时间
                self.config["last_analysis"] = current_time.isoformat()
                
                logger.info("📊 自动数据分析完成")
            
        except Exception as e:
            logger.error(f"❌ 自动数据分析失败: {e}")
    
    async def _auto_optimize(self):
        """自动优化"""
        try:
            # 检查是否需要优化
            summary = await self.data_manager.get_performance_summary()
            
            # 缓存命中率低时优化
            if summary.get("cache_hit_rate", 0) < 0.7:
                await self._optimize_cache()
            
            # 内存使用高时优化
            if summary.get("memory_usage", 0) > 512:  # 512MB
                await self._optimize_memory()
            
            # 数据库需要优化时
            if self.integration_status["data_processed"] % 100 == 0:
                await self.data_manager._optimize_database()
            
        except Exception as e:
            logger.error(f"❌ 自动优化失败: {e}")
    
    async def _optimize_cache(self):
        """优化缓存"""
        try:
            # 清理过期缓存
            await self.data_manager._cleanup_expired_data()
            
            # 调整缓存策略
            logger.info("⚡ 缓存优化完成")
            
        except Exception as e:
            logger.error(f"❌ 缓存优化失败: {e}")
    
    async def _optimize_memory(self):
        """优化内存"""
        try:
            import gc
            
            # 执行垃圾回收
            collected = gc.collect()
            
            # 清理内存缓存
            if hasattr(self.data_manager, 'memory_cache'):
                with self.data_manager.cache_lock:
                    # 保留最近使用的50%数据
                    cache_items = list(self.data_manager.memory_cache.items())
                    keep_count = len(cache_items) // 2
                    
                    # 按访问时间排序
                    cache_items.sort(key=lambda x: x[1].last_accessed, reverse=True)
                    
                    # 保留热门数据
                    self.data_manager.memory_cache = dict(cache_items[:keep_count])
            
            logger.info(f"🧠 内存优化完成，回收了 {collected} 个对象")
            
        except Exception as e:
            logger.error(f"❌ 内存优化失败: {e}")
    
    async def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            import psutil
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用
            memory = psutil.virtual_memory()
            
            # 磁盘使用
            disk = psutil.disk_usage(PROJECT_ROOT)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"❌ 收集系统指标失败: {e}")
            return None
    
    async def _collect_behavior_data(self):
        """收集行为数据"""
        try:
            # 获取活跃会话统计
            summary = await self.data_manager.get_performance_summary()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "active_sessions": summary.get("active_sessions", 0),
                "cache_hit_rate": summary.get("cache_hit_rate", 0),
                "data_reads": summary.get("performance_metrics", {}).get("data_reads", 0),
                "data_writes": summary.get("performance_metrics", {}).get("data_writes", 0)
            }
            
        except Exception as e:
            logger.error(f"❌ 收集行为数据失败: {e}")
            return None
    
    async def _collect_performance_data(self):
        """收集性能数据"""
        try:
            # 获取实时洞察
            insights = await self.data_analyzer.get_real_time_insights()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "system_status": insights.get("system_status"),
                "current_metrics": insights.get("current_metrics", {}),
                "insights_count": len(insights.get("insights", [])),
                "anomalies_count": len(insights.get("anomalies", []))
            }
            
        except Exception as e:
            logger.error(f"❌ 收集性能数据失败: {e}")
            return None
    
    async def _generate_integration_report(self):
        """生成集成报告"""
        try:
            report = {
                "integration_id": str(int(time.time())),
                "generated_at": datetime.now().isoformat(),
                "status": "active",
                "components": {
                    "data_manager": "active",
                    "data_analyzer": "active"
                },
                "statistics": {
                    "started_at": self.integration_status["started_at"].isoformat(),
                    "data_processed": self.integration_status["data_processed"],
                    "errors_count": self.integration_status["errors_count"],
                    "last_sync": self.integration_status["last_sync"].isoformat() if self.integration_status["last_sync"] else None
                },
                "configuration": self.config
            }
            
            # 保存报告
            await self.data_manager.store_data(
                data=report,
                data_type=DataType.SYSTEM_METRICS,
                priority=DataPriority.HIGH,
                tags={"integration", "report"}
            )
            
            logger.info("📋 集成报告已生成")
            
        except Exception as e:
            logger.error(f"❌ 生成集成报告失败: {e}")
    
    async def _save_integration_status(self):
        """保存集成状态"""
        try:
            status_file = PROJECT_ROOT / "data" / "integration_status.json"
            status_file.parent.mkdir(exist_ok=True)
            
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(self.integration_status, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info("💾 集成状态已保存")
            
        except Exception as e:
            logger.error(f"❌ 保存集成状态失败: {e}")

# 全局集成实例
_global_integration: Optional[ARQDataIntegration] = None

def get_integration() -> ARQDataIntegration:
    """获取全局集成实例"""
    global _global_integration
    if _global_integration is None:
        _global_integration = ARQDataIntegration()
    return _global_integration

# 信号处理
def signal_handler(signum, frame):
    """信号处理器"""
    logger.info(f"收到信号 {signum}，正在优雅停止...")
    
    integration = get_integration()
    asyncio.create_task(integration.stop_integration())

# 主函数
async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ARQ数据集成工具")
    parser.add_argument("--auto-start", action="store_true", help="自动启动集成")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--daemon", action="store_true", help="守护进程模式")
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # 创建集成实例
    integration = ARQDataIntegration(config)
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 初始化
        await integration.initialize()
        
        if args.auto_start or args.daemon:
            # 启动自动集成
            await integration.start_auto_integration()
        else:
            # 交互模式
            print("🔗 ARQ数据集成已准备就绪")
            print("可用命令:")
            print("  start    - 启动自动集成")
            print("  status   - 查看集成状态")
            print("  stop     - 停止集成")
            print("  report   - 生成报告")
            print("  exit     - 退出程序")
            
            while True:
                try:
                    command = input("\n> ").strip().lower()
                    
                    if command == "start":
                        await integration.start_auto_integration()
                    elif command == "status":
                        summary = await integration.data_manager.get_performance_summary()
                        print(f"状态: {summary}")
                    elif command == "stop":
                        await integration.stop_integration()
                    elif command == "report":
                        report = await integration.data_analyzer.generate_comprehensive_report()
                        print(f"报告ID: {report['report_id']}")
                    elif command == "exit":
                        break
                    else:
                        print("未知命令")
                        
                except EOFError:
                    break
                except Exception as e:
                    print(f"命令执行失败: {e}")
        
    except KeyboardInterrupt:
        logger.info("收到中断信号")
    except Exception as e:
        logger.error(f"程序错误: {e}")
    finally:
        # 清理资源
        await integration.stop_integration()
        logger.info("👋 ARQ数据集成已退出")

if __name__ == "__main__":
    asyncio.run(main())