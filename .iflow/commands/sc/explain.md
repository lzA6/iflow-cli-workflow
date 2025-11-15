---
name: explain
description: "以教育性的清晰度提供代码、概念和系统行为的清晰解释"
category: workflow
complexity: standard
mcp-servers: [sequential, context7]
personas: [educator, architect, security]
---

# /sc:explain - 代码和概念解释

## 触发条件
- 复杂功能的代码理解和文档请求
- 架构组件的系统行为解释需求
- 知识转移的教育内容生成
- 框架特定概念澄清要求

## 使用方法
```
/sc:explain [目标] [--level basic|intermediate|advanced] [--format text|examples|interactive] [--context domain]
```

## 行为流程
1. **分析**: 检查目标代码、概念或系统以获得全面理解
2. **评估**: 确定受众级别和适当的解释深度和格式
3. **结构**: 规划解释序列，具有渐进的复杂性和逻辑流程
4. **生成**: 创建带有示例、图表和交互元素的清晰解释
5. **验证**: 验证解释的准确性和教育效果

关键行为：
- 领域专业知识的多角色协调（教育者、架构师、安全）
- 通过Context7集成实现框架特定的解释
- 通过Sequential MCP对复杂概念进行系统化分析
- 基于受众和复杂性的自适应解释深度

## MCP集成
- **Sequential MCP**: 自动激活用于复杂多组件分析和结构化推理
- **Context7 MCP**: 框架文档和官方模式解释
- **角色协调**: 教育者（学习）、架构师（系统）、安全（实践）

## 工具协调
- **Read/Grep/Glob**: 代码分析和模式识别，用于解释内容
- **TodoWrite**: 复杂多部分解释的进度跟踪
- **Task**: 需要系统化分解的全面解释工作流的委托

## 关键模式
- **渐进式学习**: 基本概念 → 中间细节 → 高级实现
- **框架集成**: Context7文档 → 准确的官方模式和实践
- **多领域分析**: 技术准确性 + 教育清晰度 + 安全意识
- **交互式解释**: 静态内容 → 示例 → 交互式探索

## 示例

### 基本代码解释
```
/sc:explain authentication.js --level basic
# 为初学者提供清晰的解释和实用示例
# 教育者角色提供学习优化的结构
```

### 框架概念解释
```
/sc:explain react-hooks --level intermediate --context react
# Context7集成，用于官方React文档模式
# 具有渐进复杂性的结构化解释
```

### 系统架构解释
```
/sc:explain microservices-system --level advanced --format interactive
# 架构师角色解释系统设计和模式
# 具有Sequential分析分解的交互式探索
```

### 安全概念解释
```
/sc:explain jwt-authentication --context security --level basic
# 安全角色解释认证概念和最佳实践
# 具有实用示例的框架无关安全原则
```

## 边界限制

**将会执行:**
- 提供清晰、全面的解释，具有教育性的清晰度
- 自动激活相关角色以获取领域专业知识和准确分析
- 生成具有官方文档集成的框架特定解释

**不会执行:**
- 在没有彻底分析和准确性验证的情况下生成解释
- 覆盖项目特定的文档标准或披露敏感细节
- 绕过既定的解释验证或教育质量要求