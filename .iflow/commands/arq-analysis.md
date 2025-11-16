---
name: arq-analysis
description: "ARQ推理引擎 V17 Hyperdimensional Singularity"
license: Proprietary
workflow_trigger: /arq-analysis
agent_path: .iflow/agents/analysis/arq-analyzer
---

# 命令: /arq-analysis

## 核心指令：执行工作流

当此命令被触发时，**必须**执行以下Shell命令。将 `/arq-analysis` 之后的所有文本作为**位置参数**直接附加到命令末尾。

```bash
python .iflow/commands/arq-analysis-workflow-v17.py --workspace . [用户的自然语言查询]