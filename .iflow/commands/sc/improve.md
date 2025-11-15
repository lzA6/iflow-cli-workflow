---
name: improve
description: "对代码质量、性能和可维护性进行系统性改进"
category: workflow
complexity: standard
mcp-servers: [sequential, context7]
personas: [architect, performance, quality, security]
---

# /sc:improve - 代码改进

## 触发条件
- 代码质量提升和重构请求
- 性能优化和瓶颈解决需求
- 可维护性改进和技术债务削减
- 最佳实践应用和编码标准强制执行

## 使用方法
```
/sc:improve [目标] [--type quality|performance|maintainability|style] [--safe] [--interactive]
```

## 行为流程
1. **分析**: 检查代码库以寻找改进机会和质量问题
2. **计划**: 选择改进方法并激活相关角色以获取专业知识
3. **执行**: 应用系统性改进，结合领域特定的最佳实践
4. **验证**: 确保改进保留功能并符合质量标准
5. **文档**: 生成改进摘要和未来工作的建议

关键行为：
- 基于改进类型的多角色协调（架构师、性能、质量、安全）
- 通过Context7集成实现框架特定的优化，以获得最佳实践
- 通过Sequential MCP对复杂多组件改进进行系统化分析
- 具有全面验证和回滚功能的安全重构

## MCP集成
- **Sequential MCP**: 自动激活用于复杂多步改进分析和规划
- **Context7 MCP**: 框架特定的最佳实践和优化模式
- **角色协调**: 架构师（结构）、性能（速度）、质量（可维护性）、安全（安全）

## 工具协调
- **Read/Grep/Glob**: 代码分析和改进机会识别
- **Edit/MultiEdit**: 安全的代码修改和系统化重构
- **TodoWrite**: 复杂多文件改进操作的进度跟踪
- **Task**: 需要系统化协调的大规模改进工作流的委托

## 关键模式
- **质量改进**: 代码分析 → 技术债务识别 → 重构应用
- **性能优化**: 性能分析 → 瓶颈识别 → 优化实施
- **可维护性增强**: 结构分析 → 复杂性降低 → 文档改进
- **安全强化**: 漏洞分析 → 安全模式应用 → 验证验证

## 示例

### 代码质量提升
```
/sc:improve src/ --type quality --safe
# 具有安全重构应用的系统质量分析
# 改进代码结构，减少技术债务，增强可读性
```

### 性能优化
```
/sc:improve api-endpoints --type performance --interactive
# 性能角色分析瓶颈和优化机会
# 复杂性能改进决策的交互式指导
```

### 可维护性改进
```
/sc:improve legacy-modules --type maintainability --preview
# 架构师角色分析结构并建议可维护性改进
# 预览模式在应用前显示更改以供审查
```

### 安全强化
```
/sc:improve auth-service --type security --validate
# 安全角色识别漏洞并应用安全模式
# 全面验证确保安全改进有效
```

## 边界限制

**将会执行:**
- 应用系统性改进，具有领域特定的专业知识和验证
- 提供全面的分析，具有多角色协调和最佳实践
- 执行安全重构，具有回滚功能和质量保留

**不会执行:**
- 在没有适当分析和用户确认的情况下应用有风险的改进
- 在不了解完整系统影响的情况下进行架构更改
- 覆盖既定的编码标准或项目特定约定

