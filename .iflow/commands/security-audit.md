---
name: security-audit
description: "系统安全审计。当系统需要进行：(1) 安全漏洞扫描，(2) 代码安全检查，(3) 权限验证，(4) 数据安全评估，或任何安全相关审计任务时使用"
license: Proprietary. LICENSE.txt has complete terms
workflow_trigger: /security-audit
agent_path: .iflow/agents/security-auditor
---

# 系统安全审计工作流

## 功能概述

这个工作流执行全面的系统安全审计，包括漏洞扫描、代码安全检查、权限验证、数据安全评估等关键安全指标的检查。

## 执行步骤

### 1. 安全环境初始化
- 激活超级思考模式
- 初始化安全审计环境
- 加载安全检查配置

### 2. 全面安全审计
- **漏洞扫描**: OWASP Top 10漏洞检测
- **代码安全检查**: 安全编码规范验证
- **权限验证**: 访问控制和权限检查
- **数据安全评估**: 数据加密和传输安全检查

### 3. 安全加固建议
- 识别安全风险
- 提供加固策略
- 生成安全报告
- 制定改进计划

## 输入参数

```
/security-audit [可选参数]
```

**可选参数**:
- `--comprehensive`: 全面安全审计
- `--vulnerability`: 漏洞专项扫描
- `--code-security`: 代码安全检查
- `--permission`: 权限验证
- `--data-security`: 数据安全评估
- `--network`: 网络安全检查

## 输出结果

### 安全审计报告
```json
{
  "audit_summary": {
    "overall_security": "excellent/good/fair/poor",
    "vulnerability_count": 3,
    "risk_level": "low/medium/high/critical",
    "security_score": 85
  },
  "detailed_findings": {
    "vulnerability_scan": {...},
    "code_security": {...},
    "permission_analysis": {...},
    "data_security": {...}
  },
  "security_recommendations": [...],
  "remediation_plan": [...]
}
```

## 使用示例

### 基础安全审计
```
/security-audit
```

### 全面安全审计
```
/security-audit --comprehensive
```

### 漏洞专项扫描
```
/security-audit --vulnerability
```

### 代码安全检查
```
/security-audit --code-security
```

## 技术要求

- 需要访问安全审计核心模块
- 支持OWASP Top 10漏洞检测
- 需要代码静态分析能力
- 支持权限和数据安全检查

## 审计项目

### 漏洞扫描
- SQL注入检测
- XSS跨站脚本检测
- CSRF跨站请求伪造检测
- 文件包含漏洞检测
- 代码注入漏洞检测
- 不安全反序列化检测

### 代码安全检查
- 密码硬编码检查
- 敏感信息泄露检查
- 不安全API使用检查
- 错误处理安全检查
- 日志安全检查

### 权限验证
- 访问控制检查
- 权限提升检测
- 越权访问检测
- 认证机制验证

### 数据安全评估
- 数据传输加密检查
- 数据存储加密检查
- 敏感数据保护检查
- 数据备份安全检查

### 网络安全检查
- 端口安全检查
- 服务安全配置检查
- 网络隔离检查
- 防火墙配置检查

## 安全标准

### 优秀 (Excellent)
- 无高危漏洞
- 安全评分 ≥ 90分
- OWASP Top 10完全防护
- 所有安全最佳实践已实施

### 良好 (Good)
- 无中高危漏洞
- 安全评分 ≥ 80分
- 主要安全风险已控制
- 大部分安全最佳实践已实施

### 一般 (Fair)
- 存在低危漏洞
- 安全评分 ≥ 60分
- 关键安全风险已识别
- 基础安全措施已实施

### 需改进 (Poor)
- 存在高危漏洞
- 安全评分 < 60分
- 严重安全风险未处理
- 安全措施不足

## 安全检查清单

### OWASP Top 10 检查
- [ ] A01:2021 – Broken Access Control
- [ ] A02:2021 – Cryptographic Failures
- [ ] A03:2021 – Injection
- [ ] A04:2021 – Insecure Design
- [ ] A05:2021 – Security Misconfiguration
- [ ] A06:2021 – Vulnerable and Outdated Components
- [ ] A07:2021 – Identification and Authentication Failures
- [ ] A08:2021 – Software and Data Integrity Failures
- [ ] A09:2021 – Security Logging and Monitoring Failures
- [ ] A10:2021 – Server-Side Request Forgery (SSRF)

### 安全编码检查
- [ ] 输入验证和过滤
- [ ] 输出编码和转义
- [ ] 错误处理和日志记录
- [ ] 权限和访问控制
- [ ] 加密和密钥管理
- [ ] 会话管理
- [ ] 数据保护

## 注意事项

1. 安全审计可能需要10-15分钟
2. 漏洞扫描可能影响系统性能，建议在测试环境进行
3. 某些安全检查需要管理员权限
4. 发现的漏洞信息将被安全存储和处理