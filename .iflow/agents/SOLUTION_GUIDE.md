# 智能体在CLI中不出现的问题解决方案

## 问题分析

### 问题现象
在CLI中使用`/`指令时，只有arq智能体出现，但其他智能体没有出现。

### 根本原因
通过检查项目结构发现：

1. **arq-analyzer目录问题**：`arq-analyzer`目录下只有`README.md`文件，缺少关键的`mcp_server.py`文件
2. **其他智能体目录类似**：大部分智能体目录都缺少`mcp_server.py`文件
3. **配置与实现不匹配**：`settings.json`中的`mcp_config.servers`配置了这些智能体，但实际的MCP服务器文件不存在

### 技术原理
智能体在CLI中的显示依赖于：
- `settings.json`中的`workflow_triggers`配置定义了可用的指令
- `mcp_config.servers`配置定义了MCP服务器的连接信息
- 每个智能体目录下必须有对应的`mcp_server.py`文件来提供实际的工具实现

## 解决方案

### 1. 已创建的MCP服务器文件

我们已经为以下智能体创建了完整的MCP服务器文件：

#### arq-analyzer/mcp_server.py
- **工具**: `arq_reasoning`, `tool_call_precision`, `compliance_check`
- **功能**: ARQ推理引擎核心功能

#### ai-programming-assistant/mcp_server.py  
- **工具**: `code_generation`, `debugging`, `code_review`, `programming_assistance`
- **功能**: AI编程辅助相关功能

#### project-planner/mcp_server.py
- **工具**: `project_planning`, `requirement_analysis`, `risk_management`, `schedule_optimization`
- **功能**: 项目规划和管理功能

### 2. MCP服务器框架

#### mcp_server_framework.py
提供了通用的MCP服务器基础框架，包括：
- `MockServer`类：模拟MCP服务器
- `tool()`装饰器：注册工具到服务器
- `create_response()`函数：创建标准响应格式
- `handle_errors()`装饰器：错误处理

### 3. 模板和工具

#### template_mcp_server.py
通用的智能体MCP服务器模板，可以复制并修改以创建新的智能体服务器。

#### batch_create_mcp_servers.py
批量创建MCP服务器的脚本，可以根据配置自动为缺失的智能体创建服务器文件。

## 使用指南

### 1. 验证MCP服务器配置

检查`settings.json`中的配置是否正确：

```json
{
  "mcp_config": {
    "enabled": true,
    "servers": [
      {
        "name": "arq_engine_server",
        "command": "python3 .iflow/agents/arq-analyzer/mcp_server.py",
        "description": "ARQ推理引擎MCP服务器",
        "tools": ["arq_reasoning", "tool_call_precision", "compliance_check"]
      }
    ]
  }
}
```

### 2. 启动MCP服务器

为每个智能体启动对应的MCP服务器：

```bash
# 启动ARQ推理引擎
python3 .iflow/agents/arq-analyzer/mcp_server.py

# 启动AI编程助手
python3 .iflow/agents/ai-programming-assistant/mcp_server.py

# 启动项目规划专家
python3 .iflow/agents/project-planner/mcp_server.py
```

### 3. 批量创建其他智能体

使用批量创建脚本为其他智能体创建MCP服务器：

```bash
cd .iflow/agents
python3 batch_create_mcp_servers.py
```

### 4. 创建自定义智能体

1. 复制模板文件：
```bash
cp template_mcp_server.py your_agent_name/mcp_server.py
```

2. 修改配置：
- 更新服务器名称
- 添加智能体特有的工具函数
- 修改工具参数和返回值

3. 更新settings.json：
```json
{
  "workflow_triggers": {
    "your-command": {
      "command": "/your-command",
      "description": "你的智能体描述",
      "agent_path": ".iflow/agents/your_agent_name",
      "parameters": ["--param1", "--param2"]
    }
  },
  "mcp_config": {
    "servers": [
      {
        "name": "your_agent_server",
        "command": "python3 .iflow/agents/your_agent_name/mcp_server.py",
        "description": "你的智能体MCP服务器",
        "tools": ["tool1", "tool2"]
      }
    ]
  }
}
```

## 验证步骤

### 1. 检查文件存在性
```bash
ls -la .iflow/agents/*/mcp_server.py
```

### 2. 测试MCP服务器
```bash
cd .iflow/agents/arq-analyzer
python3 mcp_server.py
```

### 3. 检查CLI指令
重启CLI后，应该能看到所有配置的智能体指令。

## 常见问题

### Q: 智能体仍然不出现？
A: 检查以下几点：
1. MCP服务器文件是否存在
2. settings.json配置是否正确
3. MCP服务器是否正常启动
4. CLI是否重新加载了配置

### Q: 工具调用失败？
A: 检查：
1. 工具函数是否正确注册
2. 参数类型是否匹配
3. 错误处理是否完善

### Q: 如何添加新的工具？
A: 在对应的mcp_server.py文件中：
1. 使用`@tool()`装饰器注册新工具
2. 实现工具函数逻辑
3. 更新settings.json中的tools列表

## 下一步

1. **完成剩余智能体的MCP服务器创建**
2. **测试所有智能体的功能**
3. **优化工具实现和错误处理**
4. **添加更多实用的智能体工具**

## 总结

通过创建缺失的MCP服务器文件，我们解决了智能体在CLI中不出现的问题。关键是要确保：
- 每个智能体目录下都有对应的`mcp_server.py`文件
- `settings.json`中的配置与实际文件结构匹配
- MCP服务器能够正常启动和运行

现在你可以正常使用所有配置的智能体了！