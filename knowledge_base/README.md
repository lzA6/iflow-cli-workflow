# iFlow 知识库系统

## 📚 概述

iFlow 知识库系统是 V14 Ultra Quantum Enhanced 版本的核心功能之一，提供本地文档存储、智能索引、语义搜索和 Web UI 管理功能。系统集成了 Faiss 向量搜索引擎，支持高效的知识检索和管理。

## 🚀 核心特性

### 📊 文档管理
- **本地存储**: 所有文档安全存储在本地
- **知识组管理**: 支持创建和管理多个知识组
- **多格式支持**: 支持文本、代码等多种格式
- **自动索引**: 文档添加后自动建立向量索引

### 🔍 智能搜索
- **语义搜索**: 基于向量相似度的语义搜索
- **组内搜索**: 支持在特定知识组内搜索
- **相关性评分**: 提供文档相关性评分
- **快速检索**: 毫秒级搜索响应

### 🌐 Web UI
- **现代化界面**: 响应式设计，支持多设备
- **实时监控**: 显示系统状态和活动日志
- **便捷操作**: 直观的上传、搜索和管理界面
- **数据可视化**: 统计信息和性能指标展示

### 📈 监控与日志
- **活动日志**: 记录所有系统活动
- **性能监控**: 实时系统健康状态
- **连接追踪**: 记录所有连接和搜索请求
- **错误追踪**: 详细的错误日志和诊断

## 📁 目录结构

```
knowledge_base/
├── web_ui/              # Web UI 界面
│   ├── index.html      # 主页面
│   └── app.py          # Flask 服务器
├── groups/              # 知识组存储
│   ├── default/        # 默认组
│   └── [group_name]/   # 其他知识组
├── indexes/             # 向量索引存储
├── logs/               # 日志文件
├── config/             # 配置文件
└── README.md           # 本文档
```

## 🛠️ 安装与配置

### 系统要求
- Python 3.8+
- 2GB 可用磁盘空间
- 4GB RAM（推荐）

### 依赖安装
```bash
pip install flask flask-cors faiss-cpu numpy sentence-transformers
```

或使用 GPU 版本（需要 CUDA）：
```bash
pip install faiss-gpu
```

### 初始化
系统首次运行时会自动创建必要的目录和初始化文件。

## 🚀 快速开始

### 1. 启动 Web UI

#### Windows 用户
```bash
# 双击运行
start_knowledge_base_ui.bat

# 或命令行运行
start_knowledge_base_ui.bat
```

#### Linux/Mac 用户
```bash
python start_knowledge_base_ui.py
```

### 2. 访问 Web 界面
打开浏览器访问: http://localhost:5000

### 3. 基础使用

#### 创建知识组
1. 点击"知识组"标签
2. 输入组名和描述
3. 点击"创建知识组"

#### 上传文档
1. 点击"上传"标签
2. 填写文档标题和内容
3. 选择知识组
4. 点击"上传"

#### 搜索文档
1. 点击"搜索"标签
2. 输入搜索关键词
3. 选择知识组（可选）
4. 点击"搜索"

## 🔧 API 接口

### RESTful API

#### 获取系统统计
```http
GET /api/stats
```

#### 搜索文档
```http
POST /api/search
Content-Type: application/json

{
    "query": "搜索关键词",
    "group": "知识组名（可选）"
}
```

#### 上传文档
```http
POST /api/upload
Content-Type: application/json

{
    "title": "文档标题",
    "content": "文档内容",
    "group": "知识组名"
}
```

#### 创建知识组
```http
POST /api/groups
Content-Type: application/json

{
    "name": "知识组名",
    "description": "描述"
}
```

#### 获取日志
```http
GET /api/logs?level=info
```

#### 保存设置
```http
POST /api/settings
Content-Type: application/json

{
    "max_results": 10,
    "similarity_threshold": 0.7
}
```

## 🔗 与 ARQ 工作流集成

知识库已完全集成到 ARQ 分析工作流中：

1. **自动检索**: ARQ 分析时自动从知识库检索相关文档
2. **增强推理**: 知识库结果用于增强 ARQ 推理过程
3. **上下文提供**: 为分析提供相关的背景知识和参考资料
4. **结果引用**: 分析报告中包含知识库文档引用

### 使用示例
```bash
# 运行 ARQ 分析（会自动使用知识库）
python .iflow/commands/arq-analysis-workflow-v14.py "分析系统架构"
```

## 📊 性能优化

### 索引优化
- 使用 Faiss IVF 索引提高搜索速度
- 支持增量索引更新
- 自动索引压缩和优化

### 缓存机制
- 搜索结果缓存
- 文档向量化缓存
- 系统状态缓存

### 并发处理
- 支持多用户同时访问
- 异步文档处理
- 非阻塞搜索操作

## 🔒 安全性

### 数据安全
- 本地存储，数据不上传
- 无外部依赖，降低攻击面
- 访问日志完整记录

### 权限控制
- Web UI 仅本地访问（localhost）
- API 接口访问控制
- 文件系统权限保护

## 🐛 故障排除

### 常见问题

#### 1. 端口被占用
```bash
# 查看端口占用
netstat -ano | findstr :5000

# 修改端口
编辑 knowledge_base/web_ui/app.py，修改 port 变量
```

#### 2. 依赖安装失败
```bash
# 升级 pip
python -m pip install --upgrade pip

# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple flask flask-cors
```

#### 3. 索引损坏
```bash
# 删除索引文件（会自动重建）
del knowledge_base\indexes\*
```

#### 4. 搜索结果不准确
- 检查文档内容是否完整
- 调整相似度阈值
- 重建索引

### 日志查看
- Web UI 日志: `knowledge_base/logs/web_ui.log`
- 活动日志: `knowledge_base/logs/activity.log`
- 系统日志: 在 Web UI 的"日志"标签查看

## 📝 更新日志

### V1.0.0 (2025-11-16)
- ✅ 初始版本发布
- ✅ 基础文档管理功能
- ✅ Web UI 界面
- ✅ 与 ARQ 工作流集成
- ✅ 日志和监控系统

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

本项目采用专有许可证。

## 📞 技术支持

如有问题，请：
1. 查看本文档的故障排除部分
2. 检查系统日志
3. 通过 Issues 页面报告问题

---

**让知识管理变得简单高效！** 🚀