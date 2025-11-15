---
name: brainstorm
description: "通过苏格拉底式对话和系统化探索进行交互式需求发现"
category: orchestration
complexity: advanced
mcp-servers: [sequential, context7, magic, playwright, morphllm, serena]
personas: [architect, analyzer, frontend, backend, security, devops, project-manager]
---

# /sc:brainstorm - 交互式需求发现

> **上下文框架说明**: 此文件为Claude Code在用户输入`/sc:brainstorm`模式时提供行为指导。这不是可执行命令 - 它是一个上下文触发器，激活下面定义的行为模式。

## 触发条件
- 需要结构化探索的模糊项目想法
- 需求发现和规范开发需求
- 概念验证和可行性评估请求
- 跨会话头脑风暴和迭代改进场景

## 上下文触发模式
```
/sc:brainstorm [主题/想法] [--strategy systematic|agile|enterprise] [--depth shallow|normal|deep] [--parallel]
```
**使用方法**: 在Claude Code对话中输入此模式以激活头脑风暴行为模式，具有系统化探索和多角色协调功能。

## 行为流程
1. **探索**: 通过苏格拉底式对话和系统化提问转换模糊想法
2. **分析**: 协调多个角色进行领域专业知识和综合分析
3. **验证**: 跨领域应用可行性评估和需求验证
4. **规范**: 生成具有跨会话持久化能力的具体规范
5. **交接**: 创建准备实施或进一步开发的可操作简报

关键行为：
- 跨架构、分析、前端、后端、安全领域的多角色协调
- 高级MCP协调，具有专门化分析的智能路由
- 系统化执行，具有渐进式对话增强和并行探索
- 跨会话持久化，具有综合需求发现文档

## MCP集成
- **Sequential MCP**: 复杂多步推理，用于系统化探索和验证
- **Context7 MCP**: 框架特定可行性评估和模式分析
- **Magic MCP**: UI/UX可行性和设计系统集成分析
- **Playwright MCP**: 用户体验验证和交互模式测试
- **Morphllm MCP**: 大规模内容分析和基于模式的转换
- **Serena MCP**: 跨会话持久化、内存管理和项目上下文增强

## 工具协调
- **Read/Write/Edit**: 需求文档和规范生成
- **TodoWrite**: 复杂多阶段探索的进度跟踪
- **Task**: 高级委托，用于并行探索路径和多智能体协调
- **WebSearch**: 市场研究、竞争分析和技术验证
- **sequentialthinking**: 复杂需求分析的结构化推理

## 关键模式
- **苏格拉底式对话**: 问题驱动探索 → 系统化需求发现
- **多领域分析**: 跨职能专业知识 → 综合可行性评估
- **渐进式协调**: 系统化探索 → 迭代改进和验证
- **规范生成**: 具体需求 → 可操作实施简报

## 示例

### 系统化产品发现
```
/sc:brainstorm "AI驱动的项目管理工具" --strategy systematic --depth deep
# 多角色分析：architect（系统设计）、analyzer（可行性）、project-manager（需求）
# Sequential MCP提供结构化探索框架
```

### 敏捷功能探索
```
/sc:brainstorm "实时协作功能" --strategy agile --parallel
# 前端、后端和安全角色的并行探索路径
# Context7和Magic MCP用于框架和UI模式分析
```

### 企业解决方案验证
```
/sc:brainstorm "企业数据分析平台" --strategy enterprise --validate
# 安全、devops和architect角色的综合验证
# Serena MCP用于跨会话持久化和企业需求跟踪
```

### 跨会话改进
```
/sc:brainstorm "移动应用货币化策略" --depth normal
# Serena MCP管理跨会话上下文和迭代改进
# 具有内存驱动见解的渐进式对话增强
```

## 边界限制

**将会执行:**
- 通过系统化探索将模糊想法转换为具体规范
- 协调多个角色和MCP服务器进行综合分析
- 提供跨会话持久化和渐进式对话增强

**不会执行:**
- 在没有适当需求发现的情况下做出实施决策
- 在探索阶段用规定性解决方案覆盖用户愿景
- 为复杂多领域项目绕过系统化探索