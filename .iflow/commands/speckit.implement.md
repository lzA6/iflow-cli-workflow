---
description: 通过处理和执行 tasks.md 中定义的所有任务来执行实现计划
---

## 用户输入

```text
$ARGUMENTS
```

在继续之前，您**必须**考虑用户输入（如果不为空）。

## 大纲

1. 从仓库根目录运行 `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks` 并解析 FEATURE_DIR 和 AVAILABLE_DOCS 列表。所有路径必须是绝对路径。对于参数中的单引号，如 "I'm Groot"，请使用转义语法：例如 'I'\''m Groot'（或者如果可能的话使用双引号："I'm Groot"）。

2. **检查清单状态**（如果存在 FEATURE_DIR/checklists/）：
   - 扫描 checklists/ 目录中的所有清单文件
   - 对于每个清单，计算：
     * 总项目：所有匹配 `- [ ]` 或 `- [X]` 或 `- [x]` 的行
     * 已完成项目：匹配 `- [X]` 或 `- [x]` 的行
     * 未完成项目：匹配 `- [ ]` 的行
   - 创建状态表：
     ```
     | 清单 | 总计 | 已完成 | 未完成 | 状态 |
     |-----------|-------|-----------|------------|--------|
     | ux.md     | 12    | 12        | 0          | ✓ 通过 |
     | test.md   | 8     | 5         | 3          | ✗ 失败 |
     | security.md | 6   | 6         | 0          | ✓ 通过 |
     ```
   - 计算整体状态：
     * **通过**：所有清单都有 0 个未完成项目
     * **失败**：一个或多个清单有未完成项目
   
   - **如果任何清单未完成**：
     * 显示包含未完成项目计数的表格
     * **停止**并询问："部分清单未完成。您是否仍要继续实现？（是/否）"
     * 等待用户响应后再继续
     * 如果用户说"否"或"等待"或"停止"，暂停执行
     * 如果用户说"是"或"继续"或"进行"，继续到步骤 3
   
   - **如果所有清单都已完成**：
     * 显示显示所有清单通过的表格
     * 自动继续到步骤 3

3. 加载和分析实现上下文：
   - **必需**：读取 tasks.md 获取完整任务列表和执行计划
   - **必需**：读取 plan.md 获取技术栈、架构和文件结构
   - **如果存在**：读取 data-model.md 获取实体和关系
   - **如果存在**：读取 contracts/ 获取 API 规范和测试要求
   - **如果存在**：读取 research.md 获取技术决策和约束
   - **如果存在**：读取 quickstart.md 获取集成场景

4. **项目设置验证**：
   - **必需**：基于实际项目设置创建/验证忽略文件：
   
   **检测和创建逻辑**：
   - 检查以下命令是否成功确定仓库是否为 git 仓库（如果是则创建/验证 .gitignore）：

     ```sh
     git rev-parse --git-dir 2>/dev/null
     ```
   - 检查是否存在 Dockerfile* 或 plan.md 中有 Docker → 创建/验证 .dockerignore
   - 检查是否存在 .eslintrc* 或 eslint.config.* → 创建/验证 .eslintignore
   - 检查是否存在 .prettierrc* → 创建/验证 .prettierignore
   - 检查是否存在 .npmrc 或 package.json → 创建/验证 .npmignore（如果发布）
   - 检查是否存在 terraform 文件（*.tf）→ 创建/验证 .terraformignore
   - 检查是否需要 .helmignore（存在 helm charts）→ 创建/验证 .helmignore
   
   **如果忽略文件已存在**：验证它包含基本模式，仅追加缺失的关键模式
   **如果忽略文件缺失**：为检测到的技术创建具有完整模式集的文件
   
   **按技术的通用模式**（来自 plan.md 技术栈）：
   - **Node.js/JavaScript**: `node_modules/`、`dist/`、`build/`、`*.log`、`.env*`
   - **Python**: `__pycache__/`、`*.pyc`、`.venv/`、`venv/`、`dist/`、`*.egg-info/`
   - **Java**: `target/`、`*.class`、`*.jar`、`.gradle/`、`build/`
   - **C#/.NET**: `bin/`、`obj/`、`*.user`、`*.suo`、`packages/`
   - **Go**: `*.exe`、`*.test`、`vendor/`、`*.out`
   - **通用**: `.DS_Store`、`Thumbs.db`、`*.tmp`、`*.swp`、`.vscode/`、`.idea/`
   
   **工具特定模式**：
   - **Docker**: `node_modules/`、`.git/`、`Dockerfile*`、`.dockerignore`、`*.log*`、`.env*`、`coverage/`
   - **ESLint**: `node_modules/`、`dist/`、`build/`、`coverage/`、`*.min.js`
   - **Prettier**: `node_modules/`、`dist/`、`build/`、`coverage/`、`package-lock.json`、`yarn.lock`、`pnpm-lock.yaml`
   - **Terraform**: `.terraform/`、`*.tfstate*`、`*.tfvars`、`.terraform.lock.hcl`

5. Parse tasks.md structure and extract:
   - **Task phases**: Setup, Tests, Core, Integration, Polish
   - **Task dependencies**: Sequential vs parallel execution rules
   - **Task details**: ID, description, file paths, parallel markers [P]
   - **Execution flow**: Order and dependency requirements

6. Execute implementation following the task plan:
   - **Phase-by-phase execution**: Complete each phase before moving to the next
   - **Respect dependencies**: Run sequential tasks in order, parallel tasks [P] can run together  
   - **Follow TDD approach**: Execute test tasks before their corresponding implementation tasks
   - **File-based coordination**: Tasks affecting the same files must run sequentially
   - **Validation checkpoints**: Verify each phase completion before proceeding

7. Implementation execution rules:
   - **Setup first**: Initialize project structure, dependencies, configuration
   - **Tests before code**: If you need to write tests for contracts, entities, and integration scenarios
   - **Core development**: Implement models, services, CLI commands, endpoints
   - **Integration work**: Database connections, middleware, logging, external services
   - **Polish and validation**: Unit tests, performance optimization, documentation

8. Progress tracking and error handling:
   - Report progress after each completed task
   - Halt execution if any non-parallel task fails
   - For parallel tasks [P], continue with successful tasks, report failed ones
   - Provide clear error messages with context for debugging
   - Suggest next steps if implementation cannot proceed
   - **IMPORTANT** For completed tasks, make sure to mark the task off as [X] in the tasks file.

9. Completion validation:
   - Verify all required tasks are completed
   - Check that implemented features match the original specification
   - Validate that tests pass and coverage meets requirements
   - Confirm the implementation follows the technical plan
   - Report final status with summary of completed work

Note: This command assumes a complete task breakdown exists in tasks.md. If tasks are incomplete or missing, suggest running `/tasks` first to regenerate the task list.
