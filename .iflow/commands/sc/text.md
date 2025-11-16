# /sc:text 指令 - 交互式文本处理

## 功能描述
`/sc:text` 是一个完整的交互式文本处理指令，提供用户友好的界面让用户输入文本内容，然后执行各种文本处理操作。

## 核心功能

### 🎯 交互式界面
- **用户输入提示**: 清晰的输入引导界面
- **实时反馈**: 输入过程中的即时响应
- **选项菜单**: 多种文本处理操作选择
- **历史记录**: 保存用户的输入历史

### 🔧 文本处理功能
1. **文本分析**
   - 字数统计
   - 词频分析
   - 语言检测
   - 情感分析

2. **文本优化**
   - 格式整理
   - 错别字纠正
   - 语句优化
   - 简化/扩展

3. **格式转换**
   - Markdown转换
   - HTML转换
   - JSON格式化
   - 表格生成

4. **文本生成**
   - 摘要生成
   - 续写功能
   - 重写改写
   - 风格转换

## 使用方法

### 基础用法
```bash
/sc:text
```

### 带参数用法
```bash
# 直接指定文本内容
/sc:text --content "你的文本内容"

# 从文件读取
/sc:text --file path/to/file.txt

# 指定处理模式
/sc:text --mode analysis
/sc:text --mode optimize
/sc:text --mode convert
/sc:text --mode generate

# 输出到文件
/sc:text --output result.txt

# 交互模式
/sc:text --interactive
```

### 参数说明
- `--content`: 直接指定文本内容
- `--file`: 从文件读取文本
- `--mode`: 处理模式 (analysis/optimize/convert/generate)
- `--output`: 输出文件路径
- `--interactive`: 启用交互模式（默认）
- `--history`: 显示历史记录

## 交互式流程

### 1. 欢迎界面
```
🎯 欢迎使用 /sc:text 交互式文本处理工具

请选择操作模式：
1. 📊 文本分析
2. 🔧 文本优化
3. 🔄 格式转换
4. ✨ 文本生成
5. 📋 历史记录
6. ❌ 退出

请输入选项 (1-6):
```

### 2. 文本输入界面
```
📝 请输入您的文本内容：
支持多行输入，输入 'END' 结束

> 
```

### 3. 处理选项
```
🔧 选择处理选项：
1. 字数统计
2. 词频分析
3. 语言检测
4. 情感分析

请输入选项 (1-4):
```

### 4. 结果展示
```
📊 处理结果：
─────────────────
字数统计：156个字符
词数统计：27个词
句子数量：3个句子
段落数量：2个段落

语言：中文
情感倾向：中性
─────────────────

📁 保存结果到文件？(y/n):
```

## 实现文件

### 主要文件
- `text_enhanced_main.py` - 主入口文件
- `text_interactive.py` - 交互式界面实现
- `text_processor.py` - 文本处理核心功能
- `text_history.py` - 历史记录管理

### 配置文件
- `text_config.yaml` - 配置选项
- `text_templates.md` - 输出模板

## 启动方式

### 方法1：直接运行
```bash
python .iflow/commands/sc/text_enhanced_main.py
```

### 方法2：通过指令系统
```bash
/sc:text
```

### 方法3：带参数启动
```bash
python .iflow/commands/sc/text_enhanced_main.py --mode analysis --interactive
```

## 特色功能

### 🎨 美观界面
- 彩色输出
- 进度条显示
- 表格格式化
- 图标装饰

### 🧠 智能处理
- AI驱动的文本分析
- 上下文感知优化
- 多语言支持
- 自定义模板

### 📊 数据可视化
- 词云生成
- 统计图表
- 趋势分析
- 对比报告

### 🔒 安全保护
- 输入验证
- 数据加密
- 隐私保护
- 备份机制

## 扩展性

### 插件系统
- 自定义处理插件
- 第三方集成
- API扩展
- 模板定制

### 配置选项
- 个性化设置
- 快捷键配置
- 主题切换
- 语言选择

## 示例场景

### 1. 文档分析
```bash
/sc:text --file document.txt --mode analysis
```

### 2. 内容优化
```bash
/sc:text --content "需要优化的文本" --mode optimize
```

### 3. 格式转换
```bash
/sc:text --file notes.md --mode convert --output formatted.html
```

### 4. 摘要生成
```bash
/sc:text --file long_article.txt --mode generate --output summary.txt
```

## 更新日志

### v1.0.0 (2025-11-17)
- ✨ 初始版本发布
- 🎯 完整的交互式界面
- 🔧 基础文本处理功能
- 📊 历史记录支持
- 🎨 美观的用户界面

---

💡 **提示**: 使用 `/sc:text --help` 查看完整的帮助信息！