#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ 安全工具模块 (Security Utilities)
===================================

提供安全编码相关的工具函数：
- 安全的路径处理
- 输入验证和清理
- 安全的文件操作
- 加密和哈希工具

作者: iFlow安全团队
版本: 1.0.0
日期: 2025-11-16
"""

import os
import re
import hashlib
import secrets
from pathlib import Path
from typing import Optional, List, Any
import logging

logger = logging.getLogger(__name__)

class SecurityError(Exception):
    """安全相关异常"""
    pass

def safe_path_join(base_path: str, user_path: str) -> str:
    """
    安全的路径连接，防止路径遍历攻击
    
    Args:
        base_path: 基础路径
        user_path: 用户提供的路径
        
    Returns:
        安全的绝对路径
        
    Raises:
        SecurityError: 如果检测到路径遍历攻击
    """
    try:
        # 规范化用户路径
        normalized_user_path = os.path.normpath(user_path)
        
        # 检查危险字符
        dangerous_patterns = [
            r'\.\./',  # 向上遍历
            r'\.\.\\',  # Windows向上遍历
            r'^\.\./',  # 以向上遍历开头
            r'^\.\.\\',  # Windows以向上遍历开头
            r'^/',      # 绝对路径
            r'^\\',     # Windows绝对路径
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, normalized_user_path):
                raise SecurityError(f"检测到潜在路径遍历攻击: {user_path}")
        
        # 连接路径并规范化
        full_path = os.path.normpath(os.path.join(base_path, normalized_user_path))
        
        # 确保结果路径仍在基础路径内
        if not os.path.abspath(full_path).startswith(os.path.abspath(base_path)):
            raise SecurityError(f"路径遍历攻击被阻止: {user_path}")
        
        return full_path
        
    except Exception as e:
        if isinstance(e, SecurityError):
            raise
        logger.error(f"路径处理错误: {e}")
        raise SecurityError(f"路径处理失败: {str(e)}")

def validate_input(input_string: str, max_length: int = 1000, 
                   allowed_chars: Optional[str] = None) -> str:
    """
    验证和清理用户输入
    
    Args:
        input_string: 用户输入字符串
        max_length: 最大允许长度
        allowed_chars: 允许的字符集（正则表达式）
        
    Returns:
        清理后的安全字符串
        
    Raises:
        SecurityError: 如果输入不安全
    """
    try:
        # 检查长度
        if len(input_string) > max_length:
            raise SecurityError(f"输入长度超过限制: {len(input_string)} > {max_length}")
        
        # 检查空输入
        if not input_string.strip():
            raise SecurityError("输入不能为空")
        
        # 检查允许的字符
        if allowed_chars:
            if not re.match(f'^{allowed_chars}+$', input_string):
                raise SecurityError(f"输入包含不允许的字符: {input_string}")
        
        # 移除潜在的危险字符
        dangerous_chars = ['\0', '\r', '\n']
        cleaned = input_string
        for char in dangerous_chars:
            cleaned = cleaned.replace(char, '')
        
        return cleaned
        
    except Exception as e:
        if isinstance(e, SecurityError):
            raise
        logger.error(f"输入验证错误: {e}")
        raise SecurityError(f"输入验证失败: {str(e)}")

def safe_file_operation(file_path: str, operation: str, **kwargs) -> Any:
    """
    安全的文件操作包装器
    
    Args:
        file_path: 文件路径
        operation: 操作类型 ('read', 'write', 'append')
        **kwargs: 传递给文件操作的参数
        
    Returns:
        操作结果
        
    Raises:
        SecurityError: 如果操作不安全
    """
    try:
        path_obj = Path(file_path)
        
        # 检查文件是否存在（对于读取操作）
        if operation == 'read' and not path_obj.exists():
            raise SecurityError(f"文件不存在: {file_path}")
        
        # 检查文件大小限制
        if operation == 'read' and path_obj.exists():
            max_size = 100 * 1024 * 1024  # 100MB
            if path_obj.stat().st_size > max_size:
                raise SecurityError(f"文件过大: {path_obj.stat().st_size} > {max_size}")
        
        # 执行文件操作
        if operation == 'read':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(**kwargs)
        elif operation == 'write':
            with open(file_path, 'w', encoding='utf-8') as f:
                return f.write(**kwargs)
        elif operation == 'append':
            with open(file_path, 'a', encoding='utf-8') as f:
                return f.write(**kwargs)
        else:
            raise SecurityError(f"不支持的操作类型: {operation}")
            
    except Exception as e:
        if isinstance(e, SecurityError):
            raise
        logger.error(f"文件操作错误: {e}")
        raise SecurityError(f"文件操作失败: {str(e)}")

def generate_secure_token(length: int = 32) -> str:
    """
    生成安全的随机令牌
    
    Args:
        length: 令牌长度
        
    Returns:
        安全的随机令牌
    """
    return secrets.token_urlsafe(length)

def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    安全的密码哈希
    
    Args:
        password: 明文密码
        salt: 盐值（可选）
        
    Returns:
        (哈希值, 盐值)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    hash_obj = hashlib.pbkdf2_hmac('sha256', 
                                  password.encode('utf-8'), 
                                  salt.encode('utf-8'), 
                                  100000)
    return hash_obj.hex(), salt

def verify_password(password: str, hash_value: str, salt: str) -> bool:
    """
    验证密码
    
    Args:
        password: 明文密码
        hash_value: 哈希值
        salt: 盐值
        
    Returns:
        验证结果
    """
    computed_hash, _ = hash_password(password, salt)
    return computed_hash == hash_value

def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除危险字符
    
    Args:
        filename: 原始文件名
        
    Returns:
        安全的文件名
    """
    # 移除危险字符
    dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0']
    sanitized = filename
    
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '_')
    
    # 限制长度
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext
    
    # 确保不是空字符串
    if not sanitized or sanitized.isspace():
        sanitized = "unnamed_file"
    
    return sanitized

def validate_json_input(json_data: Any, required_fields: List[str] = None,
                       max_size: int = 1024 * 1024) -> bool:
    """
    验证JSON输入的安全性
    
    Args:
        json_data: JSON数据
        required_fields: 必需字段列表
        max_size: 最大允许大小（字节）
        
    Returns:
        验证结果
        
    Raises:
        SecurityError: 如果JSON不安全
    """
    try:
        import json
        
        # 转换为字符串检查大小
        json_str = json.dumps(json_data)
        if len(json_str.encode('utf-8')) > max_size:
            raise SecurityError(f"JSON数据过大: {len(json_str)} > {max_size}")
        
        # 检查必需字段
        if required_fields:
            if not isinstance(json_data, dict):
                raise SecurityError("JSON必须是对象类型")
            
            for field in required_fields:
                if field not in json_data:
                    raise SecurityError(f"缺少必需字段: {field}")
        
        return True
        
    except Exception as e:
        if isinstance(e, SecurityError):
            raise
        logger.error(f"JSON验证错误: {e}")
        raise SecurityError(f"JSON验证失败: {str(e)}")

# 安全配置类
class SecurityConfig:
    """安全配置"""
    
    # 输入验证配置
    MAX_INPUT_LENGTH = 1000
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_JSON_SIZE = 1024 * 1024  # 1MB
    
    # 路径安全配置
    ALLOWED_PATHS = [
        "./data",
        "./logs",
        "./temp",
        "./cache"
    ]
    
    # 密码策略
    MIN_PASSWORD_LENGTH = 8
    REQUIRE_SPECIAL_CHARS = True
    
    # 令牌配置
    TOKEN_LENGTH = 32
    TOKEN_EXPIRY = 3600  # 1小时

# 全局安全配置实例
security_config = SecurityConfig()

def is_path_allowed(path: str) -> bool:
    """
    检查路径是否在允许的路径列表中
    
    Args:
        path: 要检查的路径
        
    Returns:
        是否允许
    """
    abs_path = os.path.abspath(path)
    
    for allowed_path in security_config.ALLOWED_PATHS:
        allowed_abs = os.path.abspath(allowed_path)
        if abs_path.startswith(allowed_abs):
            return True
    
    return False

if __name__ == "__main__":
    # 测试安全工具
    print("🛡️ 安全工具模块测试")
    
    # 测试安全路径连接
    try:
        base_path = "./data"
        user_path = "../etc/passwd"  # 危险路径
        safe_path_join(base_path, user_path)
        print("❌ 路径遍历检测失败")
    except SecurityError:
        print("✅ 路径遍历检测正常")
    
    # 测试输入验证
    try:
        validate_input("test" * 300)  # 超长输入
        print("❌ 输入长度检测失败")
    except SecurityError:
        print("✅ 输入长度检测正常")
    
    # 测试令牌生成
    token = generate_secure_token()
    print(f"✅ 安全令牌生成: {token[:16]}...")
    
    print("🛡️ 安全工具模块测试完成")