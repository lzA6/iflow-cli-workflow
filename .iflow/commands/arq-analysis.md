---
name: arq-analysis
description: "V11 - 运行量子ARQ分析工作流，可根据具体查询聚焦分析，或执行全面健康检查。"
license: Proprietary
workflow_trigger: /arq-analysis
agent_path: .iflow/agents/arq-analyzer
---

# 命令: /arq-analysis

## 核心指令：执行工作流

当此命令被触发时，**必须**执行以下Shell命令。将 `/arq-analysis` 之后的所有文本作为**位置参数**直接附加到命令末尾。

```bash
python .iflow/commands/arq-analysis-workflow-v11.py --workspace . [用户的自然语言查询]