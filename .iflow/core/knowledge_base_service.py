
# 魔法数字常量定义
MAGIC_NUMBER_127_0 = 127.0
MAGIC_NUMBER_5000 = 5000
MAGIC_NUMBER_512 = 512
SECONDS_IN_MINUTE = 60
HTTP_OK = 200

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iFlow 知识库服务管理器
提供自动启动、后台运行和生命周期管理功能
"""

import os
import sys
import time
import json
import signal
import threading
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

class KnowledgeBaseService:
    """知识库服务管理器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.web_ui_dir = self.project_root / "knowledge_base" / "web_ui"
        self.service_dir = self.project_root / ".iflow" / "services"
        self.service_dir.mkdir(exist_ok=True)
        
        # 服务状态文件
        self.status_file = self.service_dir / "kb_service_status.json"
        self.pid_file = self.service_dir / "kb_service.pid"
        self.log_file = self.service_dir / "kb_service.log"
        
        # 服务配置
        self.config = {
            "host": "127.0.0.1",
            "port": 5000,
            "auto_restart": True,
            "max_memory_mb": 512,
            "health_check_interval": 30
        }
        
        self.process = None
        self.monitor_thread = None
        self.running = False
        
    def is_running(self) -> bool:
        """检查服务是否运行"""
        if not self.pid_file.exists():
            return False
            
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # 检查进程是否存在
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        status = {
            "running": self.is_running(),
            "pid": None,
            "start_time": None,
            "url": None,
            "health": "unknown"
        }
        
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    status_data = json.load(f)
                    status.update(status_data)
            except Exception as e:
                self._log(f"读取状态文件失败: {e}")
        
        if status["running"]:
            status["url"] = f"http://{self.config['host']}:{self.config['port']}"
            # 简单健康检查
            try:
                import requests
                response = requests.get(f"{status['url']}/api/health", timeout=2)
                status["health"] = "healthy" if response.status_code == 200 else "unhealthy"
            except:
                status["health"] = "unreachable"
        
        return status
    
    def start(self) -> bool:
        """启动知识库服务"""
        if self.is_running():
            self._log("知识库服务已在运行")
            return True
        
        self._log("启动知识库服务...")
        
        # 确保Web UI目录存在
        if not self.web_ui_dir.exists():
            self._log(f"错误: Web UI目录不存在: {self.web_ui_dir}")
            return False
        
        # 创建启动脚本
        start_script = self._create_start_script()
        
        try:
            # 启动子进程
            self.process = subprocess.Popen(
                [sys.executable, start_script],
                cwd=str(self.web_ui_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            # 等待服务启动
            time.sleep(2)
            
            if self.process.poll() is None:
                # 保存PID和状态
                with open(self.pid_file, 'w') as f:
                    f.write(str(self.process.pid))
                
                status_data = {
                    "pid": self.process.pid,
                    "start_time": datetime.now().isoformat(),
                    "config": self.config
                }
                
                with open(self.status_file, 'w', encoding='utf-8') as f:
                    json.dump(status_data, f, ensure_ascii=False, indent=2)
                
                self._log(f"知识库服务启动成功，PID: {self.process.pid}")
                self._log(f"访问地址: http://{self.config['host']}:{self.config['port']}")
                
                # 启动监控线程
                self.running = True
                self.monitor_thread = threading.Thread(target=self._monitor_service, daemon=True)
                self.monitor_thread.start()
                
                return True
            else:
                self._log("知识库服务启动失败")
                return False
                
        except Exception as e:
            self._log(f"启动知识库服务失败: {e}")
            return False
    
    def stop(self) -> bool:
        """停止知识库服务"""
        if not self.is_running():
            self._log("知识库服务未运行")
            return True
        
        self._log("停止知识库服务...")
        self.running = False
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # 尝试优雅关闭
            if os.name == 'nt':
                # Windows下使用taskkill
                subprocess.run(['taskkill', '/PID', str(pid), '/F'], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
            
            # 等待进程结束
            time.sleep(2)
            
            # 如果还在运行，强制终止
            if self.is_running():
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)
            
            # 清理文件
            if self.pid_file.exists():
                self.pid_file.unlink()
            if self.status_file.exists():
                self.status_file.unlink()
            
            self._log("知识库服务已停止")
            return True
            
        except Exception as e:
            self._log(f"停止知识库服务失败: {e}")
            return False
    
    def restart(self) -> bool:
        """重启知识库服务"""
        self._log("重启知识库服务...")
        self.stop()
        time.sleep(1)
        return self.start()
    
    def _create_start_script(self) -> str:
        """创建启动脚本"""
        script_path = self.service_dir / "kb_start.py"
        
        script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库服务启动脚本
"""

import os
import sys
import json
import time
from pathlib import Path

# 内存优化设置
os.environ['PYTHONOPTIMIZE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    """主函数"""
    # 导入并启动Flask应用
    sys.path.insert(0, str(project_root / "knowledge_base" / "web_ui"))
    
    # 优化Flask配置
    os.environ['FLASK_ENV'] = 'production'
    os.environ['WERKZEUG_RUN_MAIN'] = 'true'
    
    from app import app
    
    # 启动配置
    config = {{
        "host": "{self.config['host']}",
        "port": {self.config['port']},
        "debug": False,
        "threaded": True,
        "use_reloader": False
    }}
    
    try:
        app.run(**config)
    except KeyboardInterrupt:
        print("\\n知识库服务已停止")
    except Exception as e:
        print(f"知识库服务错误: {{e}}")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return str(script_path)
    
    def _monitor_service(self):
        """监控服务状态"""
        while self.running:
            time.sleep(self.config['health_check_interval'])
            
            if not self.is_running():
                self._log("检测到知识库服务异常停止")
                if self.config['auto_restart']:
                    self._log("尝试自动重启...")
                    self.start()
    
    def _log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        # 输出到控制台
        print(log_message)
        
        # 写入日志文件
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
        except Exception:
            pass
    
    def ensure_running(self) -> bool:
        """确保服务运行（如果未运行则启动）"""
        try:
            if not self.is_running():
                return self.start()
            return True
        except Exception as e:
            self._log(f"启动服务失败: {e}")
            return False

# 全局服务实例
_kb_service = None

def get_kb_service() -> KnowledgeBaseService:
    """获取知识库服务实例"""
    global _kb_service
    if _kb_service is None:
        _kb_service = KnowledgeBaseService()
    return _kb_service

def auto_start_kb_service():
    """自动启动知识库服务"""
    service = get_kb_service()
    return service.ensure_running()

def stop_kb_service():
    """停止知识库服务"""
    service = get_kb_service()
    return service.stop()

# 注册程序退出时的清理函数
import atexit
atexit.register(stop_kb_service)