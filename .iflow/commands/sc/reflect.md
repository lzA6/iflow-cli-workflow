---
name: reflect
description: "使用Serena MCP分析功能进行任务反思和验证"
category: special
complexity: standard
mcp-servers: [serena]
personas: []
---

# /sc:reflect - 任务反思和验证

## 触发条件
- 任务完成需要验证和质量评估
- 会话进度分析和对已完成工作的反思
- 跨会话学习和洞察捕获以改进项目
- 需要全面任务依从性验证的质量门

## 使用方法
```
/sc:reflect [--type task|session|completion] [--analyze] [--validate]
```

## 行为流程
1. **分析**: 使用Serena反思工具检查当前任务状态和会话进度
2. **验证**: 评估任务依从性、完成质量和需求满足度
3. **反思**: 对收集到的信息和会话洞察进行深入分析
4. **文档**: 更新会话元数据并捕获学习洞察
5. **优化**: 提供流程改进和质量提升的建议

关键行为：
- Serena MCP集成，用于全面的反思分析和任务验证
- TodoWrite模式和高级Serena分析功能之间的桥梁
- 具有跨会话持久化和学习捕获的会话生命周期集成
- 性能关键操作，核心反思和验证<200ms

## MCP集成
- **Serena MCP**: 反思分析、任务验证和会话元数据的强制集成
- **反思工具**: think_about_task_adherence, think_about_collected_information, think_about_whether_you_are_done
- **内存操作**: 具有read_memory, write_memory, list_memories的跨会话持久化
- **性能关键**: 核心反思操作<200ms，检查点创建<1s

## 工具协调
- **TodoRead/TodoWrite**: 传统任务管理和高级反思分析之间的桥梁
- **think_about_task_adherence**: 验证当前方法与项目目标和会话目标的一致性
- **think_about_collected_information**: 分析会话工作和信息收集的完整性
- **think_about_whether_you_are_done**: 评估任务完成标准和剩余工作识别
- **内存工具**: 会话元数据更新和跨会话学习捕获

## 关键模式
- **任务验证**: 当前方法 → 目标对齐 → 偏差识别 → 纠正
- **会话分析**: 信息收集 → 完整性评估 → 质量评估 → 洞察捕获
- **完成评估**: 进度评估 → 完成标准 → 剩余工作 → 决策验证
- **跨会话学习**: 反思洞察 → 内存持久化 → 增强项目理解

## 示例

### 任务依从性反思
```
/sc:reflect --type task --analyze
# 验证当前方法与项目目标的一致性
# 识别偏差并提供纠正建议
```

### 会话进度分析
```
/sc:reflect --type session --validate
# 全面分析会话工作和信息收集
# 质量评估和差距识别以改进项目
```

### 完成验证
```
/sc:reflect --type completion
# 评估任务完成标准与实际进度
# 确定任务完成的准备情况并识别剩余障碍
```

## 边界限制

**将会执行:**
- 使用Serena MCP分析工具执行全面的任务反思和验证
- 将TodoWrite模式与高级反思功能相结合，以增强任务管理
- 提供跨会话学习捕获和会话生命周期集成

**不会执行:**
- 在没有适当Serena MCP集成和反思工具访问的情况下操作
- 在没有适当依从性和质量验证的情况下覆盖任务完成决策
- 绕过会话完整性检查和跨会话持久化要求

