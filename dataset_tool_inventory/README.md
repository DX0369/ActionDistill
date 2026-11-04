# 数据集工具审查

更新时间：2026-11-04 18:11 CST +0800。

研究边界、环境和逐源许可见 [PROJECT](PROJECT.md)；当前正式扫描、分源 G3 及未完成项见 [PROJECT_STATE](PROJECT_STATE.md)。本页记录当前入口和审查方法，旧工具盘点不代替当前审查。

## 当前入口

以下正式命令由用户启动，输出必须是新目录；`--workers` 必填。默认数据根为工作区 `data/`，格式和白名单分别读取下列配置、策略文件，不按数据卡宣传补造字段。策略中的τ²污染核对路径相对于工作区根解析；显式绝对路径仍按原值读取。

```bash
conda activate mapika
cd /data/wangsq/code/ActionDistill
mkdir -p experiments/dataset_tool_inventory/outputs experiments/dataset_tool_inventory/tmp
JEV_REVIEW_NAME="jev_review_$(date +%Y%m%d_%H%M%S)"
setsid --wait python -B -u experiments/dataset_tool_inventory/scripts/review_datasets.py \
  --workers 16 \
  --config experiments/dataset_tool_inventory/configs/review_datasets.json \
  --policy experiments/dataset_tool_inventory/configs/review_policy.json \
  --tokenizer /data/wangsq/model/Mapika-decider-0.8b \
  --output "experiments/dataset_tool_inventory/outputs/$JEV_REVIEW_NAME" \
  > "experiments/dataset_tool_inventory/tmp/$JEV_REVIEW_NAME.console.log" 2>&1
```

有界 smoke 可重复指定 `--dataset`，添加 `--limit-trajectories N`，并将输出和日志改为独立 `tmp/` 位置。bounded 运行标为 partial，不用于宣布全量准入；不覆盖旧目录，也不因旧 G3 已完成就跳过新适配器的必要核查。

## 读取结果

- `review_summary.json` 是准入统计的唯一来源；`summary.json` 是基础工具计数。`review_summary.json` 的 pipeline 保留各来源 coverage；非 scanned/partial 项计数为 null，partial 只表示成功解析部分的观测计数。原始邻接统计另用 `complete_counts`，partial 时该字段为 null。
- 运行目录还保存 profiles、states/tools/issues/trajectories、重复组、污染命中、人工审查输入和长度统计；profiles 是该次运行的格式证据，不是新的 G3 裁决。正式 states 和人工审查明细仍是后续构造／分析输入，保留原路径。
- `metadata/run_status.json` 可能是 partial 或 `scan_completed_manual_review_pending`；退出码 0 不证明覆盖完整，更不代表 G3 通过。`review_samples.jsonl` 初始四项检查均为 `pending_manual_review`。
- G3 读取项目状态指定的 `manual_review/summary.json` 及 `results.jsonl`，以后审裁决和已确认的分源覆盖为准；不能只读旧逐条 NA、旧不通过或扫描里的 pending 标记。原运行状态、分数和人工原判定不回写。

## 审查方法

### 结构准入与候选来源

先定位原始记录的教师、harness、轨迹单位、动作、候选声明和结果字段，按精确配置执行白名单、多工具与当前标签检查。失败保留，未知字段保持未知；未映射教师、隔离来源和解析失败显式报告，不通过默认值或外部 schema 补齐。

| 候选来源 | 当前口径 |
|---|---|
| S | 请求捕获的结构化声明；仍需独立核查逐步可见性和候选完整性 |
| N | 恢复的 registry／snapshot；包括 README JSON 和当前 native_codex 的轨迹级 tool_schema，不冒充实时请求证据 |
| H | 无声明 harness 的有条件候选：同一 harness／领域中，在整个数据集至少 3 条轨迹调用过的工具，加本轨迹此前调用过的其他工具 |
| I | 只从本轨迹当前步之前已调用的工具恢复，完整性未证明时仅作有条件候选 |

H/I 不用本轨迹当前或未来调用扩充候选；候选生命周期未证明时不标为完整。README JSON 工具定义保留原有 parameters 及嵌套结构，不补造缺失字段。四份已配置 README 是程序输入；不要为整理文档移动、改写或删除原数据卡。

sapientai 的 airline／retail／telecom 使用 `configs/tau2_agent_tools.json` 中已批准的 τ² 官方领域工具。README 恢复只认可已批准的同数据集 snapshot，并保留有条件 N；不从外部材料补全候选。

### 历史与动作边界

- 本地轨迹单位是完整记录或独立日志文件：记录以相对文件路径与行号标识，原生日志以相对文件路径标识；原始 task/session/record ID 独立保存，不用于合并不同运行或子 agent 文件。GPT-5.6 累计前缀仍按 `source_trajectory_id` 归组，只提取 `target_message_index`。
- 当前 native_codex 将没有工具返回或可见正文隔开的连续 function_call/custom_tool_call 合为一个多调用 assistant 步骤；不跨 turn_context、压缩、用户消息或工具声明边界。不同工具名为 TOOL_SET，同名重复为 TOOL；可见正文独立。event_msg 镜像和私有 analysis/reasoning 不进历史。旧 response_item 原子步骤不是当前协议。
- Codex 整轨迹恢复的 session tool_schema 为 N，`candidate_complete=False`；它仍优先于 README，但优先级不证明实时生效区间或完整性。不因来源等级修正而回写旧样本来源字段。
- Claude Code 按 message.id 合并分片，冻结首片段前历史；沿 parentUuid 树剔除当前祖先链外被放弃的兄弟轮次及后代，保留同消息的并行 tool_use 与 tool_result。父记录缺失时不强行剔除，按 G3 已确认条件另行说明。

### G3 人工质量审查

人工核查须由样本定位回单个原始文件或对应记录，用 `review_formats.scan` 重建完整 `_history` 和 `_candidate_definitions`，按 trajectory_id、turn_index 对齐。H 候选重建时把同次 `review_summary.json` 的 `harness_tool_sets` 以 `(harness, domain)` 为键传入；动作名或候选与原 event 不同也要记录。审查表里的尾部显示或与既有 event 相同，均不能单独证明没有泄漏。每项记 pass／fail／na，并附原始行号、字段或短引用：

| 字段 | 核查内容 |
|---|---|
| `action_matches_source` | 标签来自该教师的真实动作，排除用户模拟器、judge、辅助调用和框架固定话术；同消息异名多调用为 TOOL_SET，同名多次为 TOOL，有可见正文且无调用为 RESPOND |
| `candidate_visibility` | 候选确有已批准来源，保留恢复或生命周期不完整的条件，不把并集当成每步可执行工具 |
| `no_current_or_future_leakage` | 历史不含当前动作或后续信息，Claude 真分叉与并行返回分开判断 |
| `reasoning_removed` | 教师私有推理已移除，已确认的显式 think 工具和公开字面引用正确保留 |

同时记录重复、错误循环、可疑教师动作、模型切换、用户中断、subagent 等质量标记及适用条件；既有裁决会单独指出超过 30% 的质量问题，不能通过删除失败或改口径消除。NA 不能自动视为通过或已确认泄漏；缺父记录与被放弃分支混入是不同问题。

系统性错误包括：同数据集同一种 fail 原因出现至少两次、任一当前／未来泄漏 fail，或任一标签并非教师产生的 fail。发现后再定位 2–3 个原始同类实例，记录影响范围和疑似代码位置；审查不修改适配器或历史产物。无系统性错误时，单个可解释 fail 按偶发问题记录比例；系统性错误判不通过，无系统性错误但某类质量标记超过 30% 或 H 公共集合明显不符时判有条件通过，其余按既定规则通过。先按数据集×教师判断，再汇总数据集。

当前已批准的父链复核及 Claude 重审按 [分源 G3 裁决](PROJECT_STATE.md#分源-g3-裁决)执行：旧逐条结果保留，后审覆盖原 NA；最新 Claude 的 3 条父记录缺失按用户口径四项记 pass 并注明“历史不完整”，组合仍为有条件通过。该特定裁决不扩展成对任意未知历史的自动通过。

G3 只覆盖抽中状态，不证明未抽样、排除或隔离记录正确，也不替代许可、污染、outcome、动态候选或全量覆盖审查。

### 抽样、显示与长度估算

当前审查按 policy seed=20261014 确定性抽样：每个数据集×教师模型最多 10 条准入轨迹，每条最多 3 个准入状态。人工样本输入仅显示最后 8 条历史消息，单条 JSON 超过 4,000 字符只显示前缀并记录原长；这只是显示压缩，不能用它代替 G3 的完整历史重建。

长度估算从去重前 confirmed/conditional 状态中确定性抽取最多 1,000 条，使用 Mapika-decider-0.8b tokenizer；图片替换为 `[image omitted]`，每条工具输出及每次历史工具调用参数各保留首尾 2,048 token。`length_stats.json` 在这些替换／截断之后、丢弃最早中间轮次之前计数，渲染对象是审查 JSON 加候选与 RESPOND。旧报告提到的 32,768 上限和构造阶段丢轮次不能被解释为本次长度估算已经完成最终训练截断。

## 测试与历史入口

```bash
conda activate mapika
cd /data/wangsq/code/ActionDistill/experiments/dataset_tool_inventory
python -m unittest discover -s tests
```

旧工具盘点 `build_inventory.py` 及其 registry 适配器已退役；原命令由对应 `metadata/command.txt` 定位，源码保留在原机[本地历史归档](../README.md#历史源码复现)。新输出不得覆盖历史盘点，也不能替代当前 Jev 审查。原始邻接复核脚本和结果位置见 PROJECT_STATE；邻接统计不执行 G3，也不自动改变其他格式的调用合并规则。
