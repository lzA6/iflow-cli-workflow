#!/usr/bin/env python3
"""
启动所有MCP服务器的脚本
自动启动所有配置的MCP服务器并监控状态
"""

import os
import sys
import json
import subprocess
import time
import logging
import signal
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPServerManager:
    """MCP服务器管理器"""
    
    def __init__(self, settings_file: str = ".iflow/settings.json"):
        self.settings_file = settings_file
        self.servers: Dict[str, subprocess.Popen] = {}
        self.server_configs: List[Dict[str, Any]] = []
        self.running = False
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def load_settings(self) -> Dict[str, Any]:
        """加载settings.json配置"""
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"配置文件不存在: {self.settings_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误: {e}")
            return {}
    
    def get_mcp_servers(self) -> List[Dict[str, Any]]:
        """获取所有MCP服务器配置"""
        settings = self.load_settings()
        mcp_config = settings.get("mcp_config", {})
        
        if not mcp_config.get("enabled", False):
            logger.warning("MCP配置未启用")
            return []
        
        servers = mcp_config.get("servers", [])
        logger.info(f"发现 {len(servers)} 个MCP服务器配置")
        return servers
    
    def check_server_file_exists(self, command: str) -> bool:
        """检查MCP服务器文件是否存在"""
        try:
            # 提取文件路径
            parts = command.split()
            if len(parts) >= 2:
                file_path = parts[1]
                if os.path.exists(file_path):
                    return True
                else:
                    logger.warning(f"MCP服务器文件不存在: {file_path}")
                    return False
            return False
        except Exception as e:
            logger.error(f"检查服务器文件失败: {e}")
            return False
    
    def start_server(self, server_config: Dict[str, Any]) -> Optional[subprocess.Popen]:
        """启动单个MCP服务器"""
        name = server_config.get("name", "unknown")
        command = server_config.get("command", "")
        description = server_config.get("description", "")
        
        if not self.check_server_file_exists(command):
            logger.error(f"跳过启动 {name}: 文件不存在")
            return None
        
        try:
            logger.info(f"启动MCP服务器: {name} - {description}")
            logger.info(f"命令: {command}")
            
            # 启动进程
            process = subprocess.Popen(
                command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(self.settings_file) or "."
            )
            
            self.servers[name] = process
            logger.info(f"✅ {name} 启动成功 (PID: {process.pid})")
            
            return process
            
        except Exception as e:
            logger.error(f"❌ {name} 启动失败: {e}")
            return None
    
    def start_all_servers(self):
        """启动所有MCP服务器"""
        logger.info("🚀 开始启动所有MCP服务器...")
        
        self.server_configs = self.get_mcp_servers()
        if not self.server_configs:
            logger.error("没有找到MCP服务器配置")
            return
        
        # 启动所有服务器
        for config in self.server_configs:
            self.start_server(config)
            time.sleep(1)  # 避免同时启动造成冲突
        
        self.running = True
        logger.info(f"✅ 所有MCP服务器启动完成，共 {len(self.servers)} 个服务器运行")
    
    def monitor_servers(self):
        """监控所有MCP服务器状态"""
        logger.info("🔍 开始监控MCP服务器状态...")
        
        while self.running:
            try:
                for name, process in list(self.servers.items()):
                    if process.poll() is None:
                        # 服务器正在运行
                        logger.debug(f"✅ {name} 运行正常 (PID: {process.pid})")
                    else:
                        # 服务器已停止
                        logger.warning(f"⚠️ {name} 已停止 (退出码: {process.returncode})")
                        # 尝试重启
                        config = next((c for c in self.server_configs if c.get("name") == name), None)
                        if config:
                            logger.info(f"🔄 尝试重启 {name}...")
                            new_process = self.start_server(config)
                            if new_process:
                                self.servers[name] = new_process
                
                time.sleep(10)  # 每10秒检查一次
                
            except KeyboardInterrupt:
                logger.info("收到停止信号，准备关闭所有服务器...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"监控过程中发生错误: {e}")
                time.sleep(5)
    
    def stop_all_servers(self):
        """停止所有MCP服务器"""
        logger.info("🛑 停止所有MCP服务器...")
        
        for name, process in self.servers.items():
            try:
                if process.poll() is None:
                    logger.info(f"停止 {name} (PID: {process.pid})")
                    process.terminate()
                    
                    # 等待进程结束
                    try:
                        process.wait(timeout=5)
                        logger.info(f"✅ {name} 已停止")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"强制终止 {name}")
                        process.kill()
                        process.wait()
                        
            except Exception as e:
                logger.error(f"停止 {name} 时发生错误: {e}")
        
        self.servers.clear()
        self.running = False
        logger.info("✅ 所有MCP服务器已停止")
    
    def get_server_status(self):
        """获取所有服务器状态"""
        status = {}
        for name, process in self.servers.items():
            if process.poll() is None:
                status[name] = {
                    "status": "running",
                    "pid": process.pid,
                    "uptime": "unknown"  # 可以添加更精确的运行时间计算
                }
            else:
                status[name] = {
                    "status": "stopped",
                    "exit_code": process.returncode,
                    "pid": process.pid
                }
        return status
    
    def print_status(self):
        """打印服务器状态"""
        status = self.get_server_status()
        
        print("\n" + "="*60)
        print("MCP服务器状态报告")
        print("="*60)
        
        running_count = 0
        stopped_count = 0
        
        for name, info in status.items():
            if info["status"] == "running":
                print(f"✅ {name:<25} | 运行中 (PID: {info['pid']})")
                running_count += 1
            else:
                print(f"❌ {name:<25} | 已停止 (退出码: {info['exit_code']})")
                stopped_count += 1
        
        print("-"*60)
        print(f"总计: {len(status)} 个服务器")
        print(f"运行中: {running_count}")
        print(f"已停止: {stopped_count}")
        print("="*60)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"收到信号 {signum}，准备停止所有服务器...")
        self.running = False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP服务器管理器")
    parser.add_argument("--settings", default=".iflow/settings.json", 
                       help="配置文件路径")
    parser.add_argument("--status", action="store_true",
                       help="显示服务器状态")
    parser.add_argument("--daemon", action="store_true",
                       help="以守护进程方式运行")
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.settings):
        logger.error(f"配置文件不存在: {args.settings}")
        sys.exit(1)
    
    manager = MCPServerManager(args.settings)
    
    if args.status:
        # 显示状态
        manager.print_status()
    else:
        # 启动并监控服务器
        try:
            manager.start_all_servers()
            
            if args.daemon:
                # 守护进程模式
                manager.monitor_servers()
            else:
                # 交互模式
                print("\nMCP服务器已启动，按 Ctrl+C 停止所有服务器")
                manager.monitor_servers()
                
        except KeyboardInterrupt:
            logger.info("收到停止信号，正在停止所有服务器...")
        finally:
            manager.stop_all_servers()

if __name__ == "__main__":
    main()