---
name: design
description: "设计系统架构、API和组件接口，并提供全面的规范"
category: utility
complexity: basic
mcp-servers: []
personas: []
---

# /sc:design - 系统和组件设计

## 触发条件
- 架构规划和系统设计请求
- API规范和接口设计需求
- 组件设计和技术规范要求
- 数据库模式和数据模型设计请求

## 使用方法
```
/sc:design [目标] [--type architecture|api|component|database] [--format diagram|spec|code]
```

## 行为流程
1. **分析**: 检查目标需求和现有系统上下文
2. **计划**: 根据类型和格式定义设计方法和结构
3. **设计**: 使用行业最佳实践创建全面的规范
4. **验证**: 确保设计满足需求和可维护性标准
5. **文档**: 生成带有图表和规范的清晰设计文档

关键行为：
- 需求驱动的设计方法，考虑可伸缩性
- 集成行业最佳实践以实现可维护的解决方案
- 基于需求的多格式输出（图表、规范、代码）
- 针对现有系统架构和约束进行验证

## 工具协调
- **Read**: 需求分析和现有系统检查
- **Grep/Glob**: 模式分析和系统结构调查
- **Write**: 设计文档和规范生成
- **Bash**: 需要时集成外部设计工具

## 关键模式
- **架构设计**: 需求 → 系统结构 → 可伸缩性规划
- **API设计**: 接口规范 → RESTful/GraphQL模式 → 文档
- **组件设计**: 功能需求 → 接口设计 → 实现指导
- **数据库设计**: 数据需求 → 模式设计 → 关系建模

## 示例

### 系统架构设计
```
/sc:design user-management-system --type architecture --format diagram
# 创建包含组件关系的全面系统架构
# 包括可伸缩性考虑和最佳实践
```

### API规范设计
```
/sc:design payment-api --type api --format spec
# 生成包含端点和数据模型的详细API规范
# 遵循RESTful设计原则和行业标准
```

### 组件接口设计
```
/sc:design notification-service --type component --format code
# 设计具有清晰契约和依赖关系的组件接口
# 提供实现指导和集成模式
```

### 数据库模式设计
```
/sc:design e-commerce-db --type database --format diagram
# 创建具有实体关系和约束的数据库模式
# 包括规范化和性能考虑
```

## 边界限制

**将会执行:**
- 使用行业最佳实践创建全面的设计规范
- 根据需求生成多种格式输出（图表、规范、代码）
- 针对可维护性和可伸缩性标准验证设计

**不会执行:**
- 生成实际实现代码（使用/sc:implement进行实现）
- 未经明确设计批准修改现有系统架构
- 创建违反既定架构约束的设计