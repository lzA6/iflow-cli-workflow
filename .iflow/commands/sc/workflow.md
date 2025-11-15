---
name: workflow
description: "从PRD和功能需求生成结构化实施工作流"
category: orchestration
complexity: advanced
mcp-servers: [sequential, context7, magic, playwright, morphllm, serena]
personas: [architect, analyzer, frontend, backend, security, devops, project-manager]
---

# /sc:workflow - 实施工作流生成器

## 触发条件
- PRD和功能规范分析，用于实施规划
- 开发项目的结构化工作流生成
- 复杂实施策略的多角色协调
- 跨会话工作流管理和依赖映射

## 使用方法
```
/sc:workflow [prd文件|功能描述] [--strategy systematic|agile|enterprise] [--depth shallow|normal|deep] [--parallel]
```

## 行为流程
1. **分析**: 解析PRD和功能规范以了解实施要求
2. **计划**: 生成具有依赖映射和任务编排的全面工作流结构
3. **协调**: 激活多个角色以获取领域专业知识和实施策略
4. **执行**: 创建具有自动化任务协调的结构化分步工作流
5. **验证**: 应用质量门并确保跨领域工作流的完整性

关键行为：
- 跨架构、前端、后端、安全和DevOps领域的多角色编排
- 具有智能路由的高级MCP协调，用于专业工作流分析
- 具有渐进工作流增强和并行处理的系统化执行
- 具有全面依赖跟踪的跨会话工作流管理

## MCP集成
- **Sequential MCP**: 复杂多步工作流分析和系统化实施规划
- **Context7 MCP**: 框架特定的工作流模式和实施最佳实践
- **Magic MCP**: UI/UX工作流生成和设计系统集成策略
- **Playwright MCP**: 测试工作流集成和质量保证自动化
- **Morphllm MCP**: 大规模工作流转换和基于模式的优化
- **Serena MCP**: 跨会话工作流持久化、内存管理和项目上下文

## 工具协调
- **Read/Write/Edit**: PRD分析和工作流文档生成
- **TodoWrite**: 复杂多阶段工作流执行的进度跟踪
- **Task**: 并行工作流生成和多智能体协调的高级委托
- **WebSearch**: 技术研究、框架验证和实施策略分析
- **sequentialthinking**: 复杂工作流依赖分析的结构化推理

## 关键模式
- **PRD分析**: 文档解析 → 需求提取 → 实施策略开发
- **工作流生成**: 任务分解 → 依赖映射 → 结构化实施规划
- **多领域协调**: 跨职能专业知识 → 全面实施策略
- **质量集成**: 工作流验证 → 测试策略 → 部署规划

## 示例

### 系统化PRD工作流
```
/sc:workflow ClaudeDocs/PRD/feature-spec.md --strategy systematic --depth deep
# 具有系统化工作流生成的全面PRD分析
# 多角色协调，用于完整的实施策略
```

### 敏捷功能工作流
```
/sc:workflow "用户认证系统" --strategy agile --parallel
# 具有并行任务协调的敏捷工作流生成
# Context7和Magic MCP用于框架和UI工作流模式
```

### 企业实施规划
```
/sc:workflow enterprise-prd.md --strategy enterprise --validate
# 具有全面验证的企业级工作流
# 安全、DevOps和架构师角色，用于合规性和可伸缩性
```

### 跨会话工作流管理
```
/sc:workflow project-brief.md --depth normal
# Serena MCP管理跨会话工作流上下文和持久化
# 具有内存驱动洞察的渐进工作流增强
```

## 边界限制

**将会执行:**
- 从PRD和功能规范生成全面的实施工作流
- 协调多个角色和MCP服务器，以实现完整的实施策略
- 提供跨会话工作流管理和渐进增强功能

**不会执行:**
- 执行超出工作流规划和策略的实际实施任务
- 未经适当分析和验证覆盖既定的开发流程
- 在没有全面需求分析和依赖映射的情况下生成工作流 