#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ 统一错误处理机制 (Unified Error Handling)
==============================================

提供统一的错误处理和异常管理：
- 分层异常体系
- 统一错误响应
- 错误恢复机制
- 日志和监控集成
- 用户友好的错误信息

特性：
- 结构化异常处理
- 自动错误恢复
- 错误分类和优先级
- 上下文感知的错误处理

作者: iFlow错误处理团队
版本: 1.0.0
日期: 2025-11-16
"""

import sys
import traceback
import logging
import functools
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """错误类别"""
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    BUSINESS = "business"
    EXTERNAL = "external"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """错误上下文"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

class BaseError(Exception):
    """基础异常类"""
    
    def __init__(self, 
                 message: str,
                 error_code: Optional[str] = None,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None,
                 recoverable: bool = True,
                 retry_count: int = 0):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.cause = cause
        self.recoverable = recoverable
        self.retry_count = retry_count
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'recoverable': self.recoverable,
            'retry_count': self.retry_count,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context.__dict__,
            'cause': str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"

# 具体异常类
class SystemError(BaseError):
    """系统错误"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.SYSTEM, **kwargs)

class NetworkError(BaseError):
    """网络错误"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK, **kwargs)

class DatabaseError(BaseError):
    """数据库错误"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATABASE, **kwargs)

class ValidationError(BaseError):
    """验证错误"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, recoverable=False, **kwargs)

class AuthenticationError(BaseError):
    """认证错误"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.AUTHENTICATION, recoverable=False, **kwargs)

class AuthorizationError(BaseError):
    """授权错误"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.AUTHORIZATION, recoverable=False, **kwargs)

class BusinessError(BaseError):
    """业务逻辑错误"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.BUSINESS, recoverable=False, **kwargs)

class ExternalServiceError(BaseError):
    """外部服务错误"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.EXTERNAL, **kwargs)

@dataclass
class ErrorRecoveryStrategy:
    """错误恢复策略"""
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0
    timeout: Optional[float] = None
    fallback_action: Optional[Callable] = None

class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        self.recovery_strategies: Dict[Type[BaseError], ErrorRecoveryStrategy] = {}
        self.error_callbacks: List[Callable[[BaseError], None]] = []
        self.global_recovery_strategy = ErrorRecoveryStrategy()
        
        # 默认恢复策略
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """设置默认恢复策略"""
        self.recovery_strategies.update({
            NetworkError: ErrorRecoveryStrategy(max_retries=3, retry_delay=1.0),
            DatabaseError: ErrorRecoveryStrategy(max_retries=2, retry_delay=0.5),
            ExternalServiceError: ErrorRecoveryStrategy(max_retries=2, retry_delay=2.0),
            SystemError: ErrorRecoveryStrategy(max_retries=1, retry_delay=5.0),
        })
    
    def register_recovery_strategy(self, error_type: Type[BaseError], strategy: ErrorRecoveryStrategy):
        """注册恢复策略"""
        self.recovery_strategies[error_type] = strategy
    
    def add_error_callback(self, callback: Callable[[BaseError], None]):
        """添加错误回调"""
        self.error_callbacks.append(callback)
    
    def handle_error(self, error: BaseError) -> bool:
        """处理错误"""
        try:
            # 记录错误
            self._log_error(error)
            
            # 调用错误回调
            for callback in self.error_callbacks:
                try:
                    callback(error)
                except Exception as e:
                    logger.error(f"错误回调执行失败: {e}")
            
            # 尝试恢复
            if error.recoverable:
                return self._attempt_recovery(error)
            
            return False
            
        except Exception as e:
            logger.error(f"错误处理失败: {e}")
            return False
    
    def _log_error(self, error: BaseError):
        """记录错误"""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error.severity, logging.ERROR)
        
        logger.log(log_level, f"错误发生: {error}", exc_info=error.cause)
    
    def _attempt_recovery(self, error: BaseError) -> bool:
        """尝试恢复"""
        strategy = self.recovery_strategies.get(type(error), self.global_recovery_strategy)
        
        if error.retry_count >= strategy.max_retries:
            logger.warning(f"错误重试次数已达上限: {error.retry_count}")
            return False
        
        # 等待重试延迟
        delay = strategy.retry_delay * (strategy.backoff_factor ** error.retry_count)
        
        logger.info(f"将在 {delay} 秒后重试: {error}")
        
        # 这里应该实现实际的延迟和重试逻辑
        # 由于是同步方法，这里只记录信息
        
        return True

class ErrorReporter:
    """错误报告器"""
    
    def __init__(self, report_file: Optional[str] = None):
        self.report_file = report_file or "./logs/error_reports.json"
        self.error_history: List[Dict[str, Any]] = []
        self.max_history = 1000
    
    def report_error(self, error: BaseError):
        """报告错误"""
        error_data = error.to_dict()
        
        # 添加到历史记录
        self.error_history.append(error_data)
        
        # 限制历史记录大小
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # 保存到文件
        self._save_to_file(error_data)
        
        # 检查是否需要发送警报
        self._check_alert_conditions(error_data)
    
    def _save_to_file(self, error_data: Dict[str, Any]):
        """保存到文件"""
        try:
            import json
            
            # 确保目录存在
            Path(self.report_file).parent.mkdir(parents=True, exist_ok=True)
            
            # 读取现有数据
            if Path(self.report_file).exists():
                with open(self.report_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            # 添加新错误
            existing_data.append(error_data)
            
            # 限制文件大小
            if len(existing_data) > self.max_history:
                existing_data = existing_data[-self.max_history:]
            
            # 保存文件
            with open(self.report_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"保存错误报告失败: {e}")
    
    def _check_alert_conditions(self, error_data: Dict[str, Any]):
        """检查警报条件"""
        # 严重错误警报
        if error_data['severity'] == 'critical':
            self._send_alert(f"严重错误: {error_data['message']}")
        
        # 不可恢复错误警报
        if not error_data['recoverable']:
            self._send_alert(f"不可恢复错误: {error_data['message']}")
        
        # 重试次数过多警报
        if error_data['retry_count'] > 3:
            self._send_alert(f"重试次数过多: {error_data['message']}")
    
    def _send_alert(self, message: str):
        """发送警报"""
        logger.critical(f"🚨 错误警报: {message}")
        # 这里可以集成邮件、短信、Slack等通知方式

# 全局错误处理器和报告器
_global_error_handler: Optional[ErrorHandler] = None
_global_error_reporter: Optional[ErrorReporter] = None

def get_error_handler() -> ErrorHandler:
    """获取全局错误处理器"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler

def get_error_reporter() -> ErrorReporter:
    """获取全局错误报告器"""
    global _global_error_reporter
    if _global_error_reporter is None:
        _global_error_reporter = ErrorReporter()
    return _global_error_reporter

def handle_error(error: Union[Exception, str], 
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[ErrorContext] = None) -> bool:
    """统一错误处理函数"""
    if isinstance(error, str):
        error = BaseError(error, category=category, severity=severity, context=context)
    elif not isinstance(error, BaseError):
        error = BaseError(str(error), cause=error, category=category, severity=severity, context=context)
    
    handler = get_error_handler()
    reporter = get_error_reporter()
    
    # 处理错误
    success = handler.handle_error(error)
    
    # 报告错误
    reporter.report_error(error)
    
    return success

# 装饰器
def error_handler(category: ErrorCategory = ErrorCategory.UNKNOWN,
                severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                recoverable: bool = True,
                max_retries: int = 3):
    """错误处理装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    # 创建错误上下文
                    context = ErrorContext(
                        operation=func.__name__,
                        component=func.__module__,
                        retry_count=attempt
                    )
                    
                    # 处理错误
                    if isinstance(e, BaseError):
                        e.retry_count = attempt
                        success = handle_error(e)
                    else:
                        success = handle_error(
                            e, 
                            category=category, 
                            severity=severity,
                            context=context
                        )
                    
                    # 如果不可恢复或不是最后一次尝试，继续重试
                    if not success or attempt >= max_retries:
                        break
                    
                    # 等待重试
                    delay = 1.0 * (2 ** attempt)
                    import time
                    time.sleep(delay)
            
            # 所有重试都失败，抛出最后一个错误
            raise last_error
        
        return wrapper
    return decorator

def async_error_handler(category: ErrorCategory = ErrorCategory.UNKNOWN,
                       severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                       recoverable: bool = True,
                       max_retries: int = 3):
    """异步错误处理装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    # 创建错误上下文
                    context = ErrorContext(
                        operation=func.__name__,
                        component=func.__module__,
                        retry_count=attempt
                    )
                    
                    # 处理错误
                    if isinstance(e, BaseError):
                        e.retry_count = attempt
                        success = handle_error(e)
                    else:
                        success = handle_error(
                            e, 
                            category=category, 
                            severity=severity,
                            context=context
                        )
                    
                    # 如果不可恢复或不是最后一次尝试，继续重试
                    if not success or attempt >= max_retries:
                        break
                    
                    # 等待重试
                    delay = 1.0 * (2 ** attempt)
                    await asyncio.sleep(delay)
            
            # 所有重试都失败，抛出最后一个错误
            raise last_error
        
        return wrapper
    return decorator

# 错误恢复工具
class ErrorRecovery:
    """错误恢复工具"""
    
    @staticmethod
    def safe_execute(func: Callable, *args, default=None, **kwargs):
        """安全执行函数"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            handle_error(e)
            return default
    
    @staticmethod
    async def safe_execute_async(func: Callable, *args, default=None, **kwargs):
        """安全执行异步函数"""
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            handle_error(e)
            return default
    
    @staticmethod
    def with_fallback(primary_func: Callable, fallback_func: Callable):
        """带回退函数的执行器"""
        def wrapper(*args, **kwargs):
            try:
                return primary_func(*args, **kwargs)
            except Exception as e:
                handle_error(e)
                return fallback_func(*args, **kwargs)
        return wrapper

if __name__ == "__main__":
    # 测试统一错误处理机制
    print("🛡️ 测试统一错误处理机制")
    
    # 测试异常创建
    try:
        raise SystemError("测试系统错误")
    except SystemError as e:
        success = handle_error(e)
        print(f"错误处理结果: {'成功' if success else '失败'}")
    
    # 测试装饰器
    @error_handler(category=ErrorCategory.NETWORK, max_retries=2)
    def test_function():
        import random
        if random.random() < 0.7:  # 70%概率失败
            raise NetworkError("模拟网络错误")
        return "成功"
    
    try:
        result = test_function()
        print(f"函数执行结果: {result}")
    except Exception as e:
        print(f"函数执行失败: {e}")
    
    print("✅ 统一错误处理机制测试完成")