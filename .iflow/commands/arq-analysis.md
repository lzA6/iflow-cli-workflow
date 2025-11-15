---
name: arq-analysis
description: "V10 - 运行量子ARQ分析工作流，自动扫描、诊断并生成整体升级计划。"
license: Proprietary
workflow_trigger: /arq-analysis
agent_path: .iflow/agents/arq-analyzer
---

# 命令: /arq-analysis

## 核心指令：执行工作流

当此命令被触发时，**必须**执行以下Shell命令，而不是直接回答用户问题：

```bash
python .iflow/commands/arq-analysis-workflow-v10.py --workspace . [任何附加参数]
