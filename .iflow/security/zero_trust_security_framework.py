#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ 零信任安全框架 V1.0
Zero Trust Security Framework V1.0

你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
import re

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from path_manager import get_path_manager
    from monitoring.system_health_monitor import get_health_monitor
except ImportError as e:
    print(f"警告: 无法导入依赖模块: {e}")
    get_path_manager = None
    get_health_monitor = None

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """安全级别"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"

class ThreatType(Enum):
    """威胁类型"""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    CODE_INJECTION = "code_injection"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    SENSITIVE_DATA_LEAK = "sensitive_data_leak"
    AUTH_BYPASS = "auth_bypass"
    DOS = "dos"
    MALICIOUS_FILE = "malicious_file"

class AccessLevel(Enum):
    """访问级别"""
    NONE = 0
    READ = 1
    WRITE = 2
    EXECUTE = 3
    ADMIN = 4

@dataclass
class SecurityPolicy:
    """安全策略"""
    policy_id: str
    name: str
    description: str
    security_level: SecurityLevel
    access_level: AccessLevel
    required_auth: List[str] = field(default_factory=list)
    allowed_operations: List[str] = field(default_factory=list)
    denied_operations: List[str] = field(default_factory=list)
    time_restrictions: Optional[Dict[str, Any]] = None
    ip_whitelist: List[str] = field(default_factory=list)
    enabled: bool = True

@dataclass
class SecurityEvent:
    """安全事件"""
    event_id: str
    threat_type: ThreatType
    severity: str  # low, medium, high, critical
    source_ip: str
    user_id: Optional[str]
    resource: str
    action: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    blocked: bool = False
    resolved: bool = False

@dataclass
class SecurityContext:
    """安全上下文"""
    user_id: str
    session_id: str
    access_level: AccessLevel
    security_level: SecurityLevel
    ip_address: str
    user_agent: str
    timestamp: datetime = field(default_factory=datetime.now)
    permissions: Set[str] = field(default_factory=set)
    session_data: Dict[str, Any] = field(default_factory=dict)

class ZeroTrustSecurityFramework:
    """零信任安全框架"""
    
    def __init__(self):
        """初始化零信任安全框架"""
        self.path_manager = get_path_manager() if get_path_manager else None
        self.health_monitor = get_health_monitor() if get_health_monitor else None
        
        # 安全配置
        self.security_config = {
            'encryption_key_rotation_interval': 86400,  # 24小时
            'session_timeout': 3600,  # 1小时
            'max_failed_attempts': 5,
            'lockout_duration': 900,  # 15分钟
            'audit_log_retention_days': 90,
            'real_time_monitoring': True
        }
        
        # 安全组件
        self.encryption_key = None
        self.policies = {}
        self.security_events = deque(maxlen=10000)
        self.active_sessions = {}
        self.blocked_ips = set()
        self.failed_attempts = defaultdict(int)
        
        # 威胁检测模式
        self.threat_patterns = {
            ThreatType.SQL_INJECTION: [
                r"(?i)(union|select|insert|update|delete|drop|create|alter)\s+.*\s+from",
                r"(?i)(\bor\s+1\s*=\s*1|'[^']*'\s*=\s*'[^']*')",
                r"(?i)(exec|execute)\s*\(",
                r"(?i)(sp_|xp_)\w+"
            ],
            ThreatType.XSS: [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>"
            ],
            ThreatType.PATH_TRAVERSAL: [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e\\",
                r"\.\.\/",
                r"\.\.\\"
            ],
            ThreatType.COMMAND_INJECTION: [
                r"(?i)(;|\||&|\$\(|`)",
                r"(?i)(wget|curl|nc|netcat|ssh|telnet|ftp)",
                r"(?i)(rm|mv|cp|cat|ls|ps|kill)",
                r"(?i)(/bin/|/usr/bin/|/etc/|/var/)"
            ]
        }
        
        # 初始化加密
        self._initialize_encryption()
        
        # 加载安全策略
        self._load_security_policies()
        
        # 设置日志
        self._setup_logging()
        
        logger.info("🛡️ 零信任安全框架初始化完成")
    
    def _initialize_encryption(self):
        """初始化加密"""
        # 生成加密密钥
        password = secrets.token_bytes(32)
        salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        
        self.encryption_key = Fernet(key)
        self.encryption_salt = salt
        
        logger.info("🔐 加密系统初始化完成")
    
    def _load_security_policies(self):
        """加载安全策略"""
        # 默认安全策略
        default_policies = [
            SecurityPolicy(
                policy_id="public_access",
                name="公共访问策略",
                description="允许公共资源的只读访问",
                security_level=SecurityLevel.PUBLIC,
                access_level=AccessLevel.READ,
                allowed_operations=["GET", "HEAD", "OPTIONS"],
                denied_operations=["POST", "PUT", "DELETE", "PATCH"]
            ),
            SecurityPolicy(
                policy_id="internal_access",
                name="内部访问策略",
                description="内部用户的完全访问权限",
                security_level=SecurityLevel.INTERNAL,
                access_level=AccessLevel.WRITE,
                required_auth=["session", "mfa"],
                allowed_operations=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
            ),
            SecurityPolicy(
                policy_id="admin_access",
                name="管理员访问策略",
                description="管理员完全控制权限",
                security_level=SecurityLevel.SECRET,
                access_level=AccessLevel.ADMIN,
                required_auth=["session", "mfa", "admin_token"],
                allowed_operations=["ALL"]
            )
        ]
        
        for policy in default_policies:
            self.policies[policy.policy_id] = policy
        
        logger.info(f"📋 已加载 {len(self.policies)} 个安全策略")
    
    def _setup_logging(self):
        """设置安全日志"""
        if not self.path_manager:
            return
        
        log_dir = self.path_manager.log_dir
        log_dir.mkdir(exist_ok=True)
        
        # 安全日志文件
        security_log_file = log_dir / f"security_{datetime.now().strftime('%Y%m%d')}.log"
        
        security_logger = logging.getLogger("security_framework")
        security_logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(security_log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        security_logger.addHandler(file_handler)
        self.security_logger = security_logger
    
    async def authenticate(self, user_id: str, credentials: Dict[str, Any], 
                          context: Dict[str, Any]) -> Optional[SecurityContext]:
        """用户认证"""
        try:
            # 检查失败尝试
            ip_address = context.get('ip_address', 'unknown')
            if self.failed_attempts[user_id] >= self.security_config['max_failed_attempts']:
                self._block_ip(ip_address)
                self._log_security_event(
                    ThreatType.AUTH_BYPASS,
                    "high",
                    ip_address,
                    user_id,
                    "authentication",
                    "max_failed_attempts_exceeded"
                )
                return None
            
            # 验证凭据（简化实现）
            if not self._verify_credentials(user_id, credentials):
                self.failed_attempts[user_id] += 1
                self._log_security_event(
                    ThreatType.AUTH_BYPASS,
                    "medium",
                    ip_address,
                    user_id,
                    "authentication",
                    "invalid_credentials"
                )
                return None
            
            # 重置失败计数
            self.failed_attempts[user_id] = 0
            
            # 创建安全上下文
            session_id = secrets.token_urlsafe(32)
            security_context = SecurityContext(
                user_id=user_id,
                session_id=session_id,
                access_level=self._determine_access_level(user_id),
                security_level=self._determine_security_level(user_id),
                ip_address=ip_address,
                user_agent=context.get('user_agent', ''),
                permissions=self._get_user_permissions(user_id)
            )
            
            # 存储会话
            self.active_sessions[session_id] = security_context
            
            self.security_logger.info(f"✅ 用户认证成功: {user_id}")
            
            return security_context
        
        except Exception as e:
            self.security_logger.error(f"认证过程中出错: {e}")
            return None
    
    def _verify_credentials(self, user_id: str, credentials: Dict[str, Any]) -> bool:
        """验证凭据（简化实现）"""
        # 在实际实现中，这里应该连接到认证数据库
        # 现在只是简单的演示
        password = credentials.get('password', '')
        
        # 模拟密码验证
        if user_id == "admin" and password == "admin123":
            return True
        elif user_id == "user" and password == "user123":
            return True
        
        return False
    
    def _determine_access_level(self, user_id: str) -> AccessLevel:
        """确定访问级别"""
        if user_id == "admin":
            return AccessLevel.ADMIN
        elif user_id.startswith("user_"):
            return AccessLevel.WRITE
        else:
            return AccessLevel.READ
    
    def _determine_security_level(self, user_id: str) -> SecurityLevel:
        """确定安全级别"""
        if user_id == "admin":
            return SecurityLevel.SECRET
        elif user_id.startswith("user_"):
            return SecurityLevel.INTERNAL
        else:
            return SecurityLevel.PUBLIC
    
    def _get_user_permissions(self, user_id: str) -> Set[str]:
        """获取用户权限"""
        if user_id == "admin":
            return {"read", "write", "execute", "admin"}
        elif user_id.startswith("user_"):
            return {"read", "write"}
        else:
            return {"read"}
    
    async def authorize(self, security_context: SecurityContext, 
                      resource: str, action: str) -> bool:
        """授权检查"""
        try:
            # 检查会话有效性
            if not self._is_session_valid(security_context):
                return False
            
            # 检查IP白名单
            if not self._check_ip_whitelist(security_context):
                self._log_security_event(
                    ThreatType.AUTH_BYPASS,
                    "medium",
                    security_context.ip_address,
                    security_context.user_id,
                    "authorization",
                    "ip_not_whitelisted"
                )
                return False
            
            # 检查权限
            if not self._check_permissions(security_context, resource, action):
                self._log_security_event(
                    ThreatType.AUTH_BYPASS,
                    "medium",
                    security_context.ip_address,
                    security_context.user_id,
                    "authorization",
                    f"insufficient_permissions_for_{action}"
                )
                return False
            
            # 检查时间限制
            if not self._check_time_restrictions(security_context):
                self._log_security_event(
                    ThreatType.AUTH_BYPASS,
                    "low",
                    security_context.ip_address,
                    security_context.user_id,
                    "authorization",
                    "time_restriction_violation"
                )
                return False
            
            return True
        
        except Exception as e:
            self.security_logger.error(f"授权检查时出错: {e}")
            return False
    
    def _is_session_valid(self, security_context: SecurityContext) -> bool:
        """检查会话有效性"""
        session_timeout = self.security_config['session_timeout']
        
        if datetime.now() - security_context.timestamp > timedelta(seconds=session_timeout):
            # 清理过期会话
            if security_context.session_id in self.active_sessions:
                del self.active_sessions[security_context.session_id]
            return False
        
        return security_context.session_id in self.active_sessions
    
    def _check_ip_whitelist(self, security_context: SecurityContext) -> bool:
        """检查IP白名单"""
        # 检查是否被阻止
        if security_context.ip_address in self.blocked_ips:
            return False
        
        # 检查策略中的IP白名单
        for policy in self.policies.values():
            if (policy.enabled and 
                policy.ip_whitelist and 
                security_context.ip_address not in policy.ip_whitelist):
                return False
        
        return True
    
    def _check_permissions(self, security_context: SecurityContext, 
                          resource: str, action: str) -> bool:
        """检查权限"""
        # 检查访问级别
        required_level = self._get_required_access_level(action)
        if security_context.access_level.value < required_level:
            return False
        
        # 检查具体权限
        if action.lower() in ["read", "get", "list"]:
            return "read" in security_context.permissions
        elif action.lower() in ["write", "create", "update", "delete"]:
            return "write" in security_context.permissions
        elif action.lower() in ["execute", "run", "admin"]:
            return "execute" in security_context.permissions or "admin" in security_context.permissions
        
        return True
    
    def _get_required_access_level(self, action: str) -> int:
        """获取操作所需的访问级别"""
        read_actions = ["read", "get", "list", "head", "options"]
        write_actions = ["write", "create", "update", "post", "put", "patch"]
        execute_actions = ["execute", "run", "delete", "admin"]
        
        if action.lower() in read_actions:
            return AccessLevel.READ.value
        elif action.lower() in write_actions:
            return AccessLevel.WRITE.value
        elif action.lower() in execute_actions:
            return AccessLevel.EXECUTE.value
        else:
            return AccessLevel.ADMIN.value
    
    def _check_time_restrictions(self, security_context: SecurityContext) -> bool:
        """检查时间限制"""
        current_hour = datetime.now().hour
        
        for policy in self.policies.values():
            if (policy.enabled and 
                policy.time_restrictions and 
                security_context.security_level == policy.security_level):
                
                restrictions = policy.time_restrictions
                allowed_hours = restrictions.get('allowed_hours', [])
                
                if allowed_hours and current_hour not in allowed_hours:
                    return False
        
        return True
    
    async def scan_for_threats(self, data: str, context: Dict[str, Any]) -> List[SecurityEvent]:
        """扫描威胁"""
        threats = []
        
        try:
            for threat_type, patterns in self.threat_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, data, re.IGNORECASE)
                    
                    for match in matches:
                        threat = SecurityEvent(
                            event_id=secrets.token_urlsafe(16),
                            threat_type=threat_type,
                            severity=self._determine_threat_severity(threat_type, match.group()),
                            source_ip=context.get('ip_address', 'unknown'),
                            user_id=context.get('user_id'),
                            resource=context.get('resource', 'unknown'),
                            action="threat_detected",
                            details={
                                'pattern': pattern,
                                'match': match.group(),
                                'position': match.span()
                            }
                        )
                        
                        threats.append(threat)
            
            # 记录威胁事件
            for threat in threats:
                self.security_events.append(threat)
                self._log_security_event(
                    threat.threat_type,
                    threat.severity,
                    threat.source_ip,
                    threat.user_id,
                    threat.action,
                    threat.details.get('match', 'unknown')
                )
        
        except Exception as e:
            self.security_logger.error(f"威胁扫描时出错: {e}")
        
        return threats
    
    def _determine_threat_severity(self, threat_type: ThreatType, match: str) -> str:
        """确定威胁严重性"""
        high_severity_patterns = [
            "drop table", "exec(", "system(", "<?php", "<script",
            "../..", "rm -rf", "wget ", "curl "
        ]
        
        for pattern in high_severity_patterns:
            if pattern.lower() in match.lower():
                return "critical"
        
        if threat_type in [ThreatType.SQL_INJECTION, ThreatType.CODE_INJECTION, ThreatType.COMMAND_INJECTION]:
            return "high"
        elif threat_type in [ThreatType.XSS, ThreatType.CSRF, ThreatType.PATH_TRAVERSAL]:
            return "medium"
        else:
            return "low"
    
    def _block_ip(self, ip_address: str):
        """阻止IP"""
        self.blocked_ips.add(ip_address)
        self.security_logger.warning(f"🚫 IP已阻止: {ip_address}")
    
    def _log_security_event(self, threat_type: ThreatType, severity: str,
                           source_ip: str, user_id: Optional[str],
                           resource: str, action: str, details: str = ""):
        """记录安全事件"""
        event = SecurityEvent(
            event_id=secrets.token_urlsafe(16),
            threat_type=threat_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            resource=resource,
            action=action,
            details={"description": details}
        )
        
        self.security_events.append(event)
        
        # 记录到日志
        severity_icon = {"low": "ℹ️", "medium": "⚠️", "high": "🚨", "critical": "🔴"}
        icon = severity_icon.get(severity, "📢")
        
        self.security_logger.warning(
            f"{icon} 安全事件: {threat_type.value} - {severity} - "
            f"IP: {source_ip} - 用户: {user_id} - 资源: {resource} - {details}"
        )
    
    def encrypt_data(self, data: str) -> str:
        """加密数据"""
        if not self.encryption_key:
            raise RuntimeError("加密系统未初始化")
        
        encrypted_data = self.encryption_key.encrypt(data.encode())
        return encrypted_data.decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """解密数据"""
        if not self.encryption_key:
            raise RuntimeError("加密系统未初始化")
        
        decrypted_data = self.encryption_key.decrypt(encrypted_data.encode())
        return decrypted_data.decode()
    
    def generate_api_key(self, user_id: str, permissions: List[str]) -> str:
        """生成API密钥"""
        timestamp = int(time.time())
        random_part = secrets.token_urlsafe(16)
        
        api_key_data = f"{user_id}:{timestamp}:{','.join(permissions)}:{random_part}"
        api_key = self.encrypt_data(api_key_data)
        
        # 移除加密数据的特殊字符，使其适合作为API密钥
        api_key = api_key.replace('+', '-').replace('/', '_').replace('=', '')
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """验证API密钥"""
        try:
            # 还原API密钥格式
            restored_key = api_key.replace('-', '+').replace('_', '/').rstrip('=')
            
            # 添加填充
            padding = len(restored_key) % 4
            if padding:
                restored_key += '=' * (4 - padding)
            
            decrypted_data = self.decrypt_data(restored_key)
            
            parts = decrypted_data.split(':')
            if len(parts) != 4:
                return None
            
            user_id = parts[0]
            timestamp = int(parts[1])
            permissions = parts[2].split(',')
            random_part = parts[3]
            
            # 检查时间戳（API密钥有效期30天）
            if time.time() - timestamp > 30 * 24 * 3600:
                return None
            
            return {
                'user_id': user_id,
                'permissions': permissions,
                'timestamp': timestamp,
                'random_part': random_part
            }
        
        except Exception as e:
            self.security_logger.error(f"API密钥验证失败: {e}")
            return None
    
    def get_security_status(self) -> Dict[str, Any]:
        """获取安全状态"""
        recent_events = list(self.security_events)[-100:]  # 最近100个事件
        
        # 统计威胁类型
        threat_stats = defaultdict(int)
        severity_stats = defaultdict(int)
        
        for event in recent_events:
            threat_stats[event.threat_type.value] += 1
            severity_stats[event.severity] += 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'active_sessions': len(self.active_sessions),
            'blocked_ips': len(self.blocked_ips),
            'total_events': len(self.security_events),
            'recent_events': len(recent_events),
            'threat_statistics': dict(threat_stats),
            'severity_statistics': dict(severity_stats),
            'security_policies': len(self.policies),
            'encryption_active': self.encryption_key is not None
        }

# 全局安全框架实例
_zero_trust_framework = None

def get_zero_trust_framework() -> ZeroTrustSecurityFramework:
    """获取全局零信任安全框架实例"""
    global _zero_trust_framework
    if _zero_trust_framework is None:
        _zero_trust_framework = ZeroTrustSecurityFramework()
    return _zero_trust_framework

async def main():
    """主函数 - 零信任安全框架测试"""
    framework = get_zero_trust_framework()
    
    print("🛡️ 启动零信任安全框架测试...")
    
    # 测试认证
    print("\n🔐 测试用户认证...")
    context = {
        'ip_address': '192.168.1.100',
        'user_agent': 'Test-Agent/1.0'
    }
    
    # 认证管理员
    admin_context = await framework.authenticate(
        "admin", 
        {"password": "admin123"}, 
        context
    )
    
    if admin_context:
        print(f"✅ 管理员认证成功: {admin_context.session_id}")
        
        # 测试授权
        authorized = await framework.authorize(
            admin_context, 
            "system_config", 
            "read"
        )
        print(f"📋 授权检查: {'通过' if authorized else '拒绝'}")
    
    # 测试威胁扫描
    print("\n🔍 测试威胁扫描...")
    test_data = "SELECT * FROM users WHERE id = 1 OR '1'='1'; <script>alert('xss')</script>"
    threats = await framework.scan_for_threats(test_data, context)
    print(f"🚨 发现威胁: {len(threats)} 个")
    
    for threat in threats:
        print(f"  - {threat.threat_type.value}: {threat.severity}")
    
    # 获取安全状态
    status = framework.get_security_status()
    print("\n📊 安全状态:")
    print(json.dumps(status, indent=2, default=str))

if __name__ == "__main__":
    import base64
    asyncio.run(main())