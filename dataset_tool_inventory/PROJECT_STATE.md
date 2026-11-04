# 数据集审查状态

更新时间：2026-11-04 14:30 CST +0800。

## 当前阶段

当前维护审查入口、共享聚合和 schema；旧工具盘点 CLI、registry 及专用适配器已经退役，历史源码保留在原机本地归档，不随公开主线发布。当前语义和逐源许可见 [PROJECT](PROJECT.md)，操作与精确审查方法见 [README](README.md)。本轮将τ²污染核对路径改为相对工作区解析，保留显式绝对路径，30项CPU回归通过；没有正式扫描、G3重审、样本构造、训练或模型／API请求，原始数据卡、快照、许可证及正式证据不改写。

native_codex 连续调用边界及 τ² 缺失身份污染标记已修复，并完成有界构造验证；最近正式扫描仍早于 Codex 合并修复。GPT-5.5-Codex、gpt-5.5-agent、Kimi-K3-Codex 的旧决策点数量不能作为新适配器计数。Claude 修复后 G3 重审已经完成；Codex 修复后的新全量审查尚未进行，是另一项未完成工作。

## 当前审查与证据

| 用途 | 项目相对路径／状态 |
|---|---|
| 最新正式扫描 | `outputs/jev_review_20261015_005012/`；partial，4 组隔离、6 个数据集解析失败 |
| Claude 当前 G3 | 最新扫描的 `manual_review/summary.json` 和 `results.jsonl`，只覆盖 claude-fable-5 的 30 个状态 |
| 其余来源 G3 | `outputs/jev_review_20261014_231124/manual_review/summary.json` 和 `results.jsonl`；沿用其余 22 个数据集，`post_review_adjudication` 优先于旧逐条 NA 及旧原因文字 |
| 原始邻接统计 | `outputs/call_adjacency_20261015_171519/summary.json`；partial |
| 邻接复核入口 | `tmp/call_adjacency_20261015/count_call_adjacency.py`，保留原位 |

两份正式审查的 states 和 G3 明细仍被 Jev 构造、教师循环及 BFCL 分析读取，全部保留原路径。原扫描的 pending 人工标记、旧逐条判定及旧不通过均不回写；不能用它们覆盖后来明确分源的裁决。

### 分源 G3 裁决

当前合并结论为 **18 个数据集通过、5 个有条件通过**。旧 G3 独立审查 2,147 个唯一状态、76 个数据集×教师组合、23 个有准入样本的数据集；最新重审只替换其中 claude-fable-5 的 30 条，不重审其他来源。G3 通过不代表训练许可、动态工具生命周期、全量覆盖或未抽样状态已经验证。

下表“旧”指 `jev_review_20261014_231124/manual_review/results.jsonl`，“新”指 `jev_review_20261015_005012/manual_review/results.jsonl`；行号是逐条定位入口，最终条件读取对应 summary 及后审裁决。表中 Codex 的通过是历史样本 G3，不是对当前合并协议或实时请求来源的新验证。

| 数据集 | 抽中状态 | G3 | 保留条件／逐条定位 |
|---|---:|---|---|
| hermes-agent-reasoning-traces | 58 | 通过 | GLM 两状态重复 todo({}) 无进展，质量标记未超过 30%；旧:1 |
| WildClawBench-Trajectories | 234 | 通过 | 四项检查通过，不替代候选生命周期核查；旧:59 |
| Toolathlon-Trajectories | 713 | 通过 | 保留 5 个可疑教师动作、重复调用等质量标记，均未超过 30%；旧:293 |
| sapientai-opus-5-tau2-trajectories | 30 | 通过 | 仅该批抽中状态；旧:1006 |
| deepseek-v4-pro-0813-agentic | 29 | 通过 | 仅该批抽中状态；旧:1036 |
| ZClawBench | 172 | 有条件通过 | H 集合符合规则，但存在缺凭证、禁用、断连或工具不可用；MiniMax M2.5 错误循环 9/27（33.3%），保留环境及质量条件；旧:1065 |
| finance-agents-benchmark-traces | 120 | 通过 | 仅该批抽中状态；旧:1237 |
| claude-fable-5-claude-code-traces | 30 | 有条件通过 | 修复后四项各 30 pass；5 个分支剔除状态核查正确，3 个父记录缺失；新:7、22 及 summary |
| Fable-5-Cursor-Traces | 30 | 通过 | 仅该批抽中状态，N 来源条件不变；旧:1387 |
| fable-5-claude-code-traces | 31 | 有条件通过 | 26 个原 NA 为并行返回／行序问题，后审改判通过；H 候选仍有条件；旧:1417 |
| mimo-claude-code-traces-1k | 27 | 通过 | 仅该批抽中状态；旧:1448 |
| gpt-5.5-agent | 45 | 通过 | 不验证修复后的新计数或候选可见时段；旧:1475 |
| GLM-5.2-Agent | 60 | 通过 | 同数据集 README snapshot，N 候选条件不变；旧:1520 |
| Kimi-K3-Codex | 15 | 通过 | 同轨迹尾部恢复 schema，不是实时请求捕获；当前来源为 N／不完整；旧:1580 |
| kimi-k2.6-claude-code-traces | 30 | 有条件通过 | 25 个原 NA 为父记录缺失，历史可能不完整，不判为未来信息泄漏；旧:1595 |
| minimax-m3-claude-code-traces | 30 | 有条件通过 | 9 个原 NA 为并行返回分叉，后审改判通过；H 候选仍有条件；旧:1637 |
| swe-rebench-v2-glm-5.1-pi-traces | 30 | 通过 | 仅该批抽中状态；旧:1655 |
| Open-SWE-Traces | 30 | 通过 | 仅该批抽中状态；旧:1685 |
| unified-agent-trajectories | 285 | 通过 | 许可和恢复候选条件仍未消除；旧:1715 |
| DeepSeek-v4-Pro-Agent | 28 | 通过 | 许可及候选完整性条件仍未消除；旧:2000 |
| GPT-5.5-Codex | 30 | 通过 | 同轨迹尾部恢复 schema，不是实时请求捕获；当前来源为 N／不完整；旧:2028 |
| gpt-5.6-sol-coding-and-debugging-traces | 30 | 通过 | 仅该批 target_message_index 状态；旧:2058 |
| mle-trajectories | 60 | 通过 | 仅该批抽中状态，outcome 等未知不被 G3 补齐；旧:2088 |

旧 G3 的 72 个可见性 NA 后审分为 39 个正常并行／行序问题、28 个父记录缺失、5 个被放弃分支混入；前者改判、缺父记录保留条件，最后一类确认了旧 native_claude 缺陷。Claude 修复后的新 G3 以最新 summary 的 `branch_removal_reviews`、`incomplete_history_states`、`inherited_g3` 为准，其余来源以旧 summary 的 `post_review_adjudication` 为准，不把旧结论直接套到修复后记录。

Claude 新审查的 5 个状态保留了重发版本和全部应见并行返回；3 个不完整状态来自 `c6d4788b-6bb6-4774-8294-5dc897346ca9.jsonl` 的 42／150／180 行，祖先 UUID `04a85be1-fd93-4d39-a1a5-6c2cc96428d9` 缺失。按用户确认，四项记 pass 并注明“历史不完整（父记录缺失）”，不是 NA，组合仍有条件。质量标记 user_interrupt=8/30、subagent=3/30，均未超过 30%；51 工具 README snapshot 仍为有条件 N。

### Codex 来源与可见时段

当前 native_codex 将整条日志里的 session tool_schema 记作 N 级恢复材料、候选完整性为 false，保留 session 优先于 README。旧产物里的 S 字段仍是历史值，不能作为实时请求捕获已验证的证据。

有界原始核查中，GPT-5.5-Codex、Kimi 各两例的 schema 都在 task_complete 后追加，仅有 source=teich，无请求关联或生效区间；两来源各有一例实际调用 write_stdin 而 session 表缺失，Kimi README 含此工具，GPT README 也缺失。gpt-5.5-agent 两例无 tool_schema，README 与实际调用不一致。这支持 N／不完整条件，不能证明具体恢复机制；本地未找到对应导出版本实现。下一次正式构造前仍需审阅缺失工具、session／README 差异和可见时段；本轮不补候选、不改优先级，也不据此断言已发生或排除泄漏。

## 已确认结论与验证范围

- 原始邻接统计为 partial：22 数据集完整、2 部分、11 隔离。Hermes 有 22 条推理块未闭合，Toolathlon 有 113 条 messages 非列表；失败记录整体不计，部分覆盖不冒充全量零。
- 严格相邻调用对（A/B/C 算 2 对）：native_codex 7,726（GPT-5.5-Codex 6,730、gpt-5.5-agent 994、Kimi 2）；native_claude 553（同 message.id 分片 507、其他 46）；native_pi 76（DeepSeek 62、GLM 14）；openai 3（unified 1、GPT-5.6 目标前缀 2）。未套教师准入过滤或恢复 Claude 分支，不能从邻接直接推断应合并；非 Codex 的解析规则未自行修改。
- Codex 合并后有界构造：每数据集前 20 条输入、每轨迹首中末代表状态，共 1,107 条；train/val/test_id/test_ood 为 912/55/22/118。长度排除 0，ID 唯一、split 组隔离及分布检查通过。该运行是 partial，不代表全量审查或正式构造；代表样本 TOOL_SET=0 不代表全量为零，异名合并由边界测试验证。
- τ² 污染只将缺失 domain/task_id 标为 unknown；明确非 retail 或不在测试集的身份仍 clear。Jev build_report 支持按 harness 的动作类型和候选等级分布。

## 简要历程与待解决事项

- 2026-10-15 10:05 CST +0800：在早期工具盘点和 35 项审查配置、H 候选及分源 G3 基础上，完成 Claude 分支修复后的 30 条重审；保留其余 22 来源旧 G3 及后审条件。旧报告里的早期 13 项优先级、被取代扫描统计、1000–2000 状态重采集和 W/S/T1/TR 在线对照属于历史设想，不是当前待运行协议。
- 2026-11-02 04:45 PST -0800：修复 README JSON 快照读取，保留原 parameters 和嵌套定义，不补字段。27 项审查、2 项训练渲染边界 CPU 测试通过；四份真实 README 共 70 工具逐项保留，四个原始记录完成有界渲染且未新增历史截断。没有重跑全量审查、G3 或构造。
- 2026-11-02 08:02 PST -0800：将 Codex 整轨迹 session tool_schema 来源改为 N／不完整，候选内容和顺序不变。27 项审查、21 项 Jev 构造 CPU 测试通过；三个 Codex 来源首末共 6 条日志对照修复前提交 4d7d8cb 的 650 个决策点，538 个 S 改 N，其他事件和准入逐项一致；18 个代表状态的渲染、token IDs、顺序及 gold 一致。该 smoke 未重算全量 harness 并集、正式构造或科研指标，临时材料已清理。
- 2026-11-04 11:10 CST +0800：退役旧盘点 CLI、registry 及专用适配器，保留当前 review 和共享组件；ToolACE 三项常量原样归入 review_formats，GPT 前缀测试改验当前审查。29 项审查／共享组件、21 项 Jev 构造 CPU 测试通过；两份旧 inventory 问题明细、正式 review/G3 及邻接证据保留。

后续仍需按授权处理：Codex 修复后的新全量审查；动态工具生命周期和候选完整性；PROJECT 许可表中的逐源未决条款；污染及部分 outcome 未知。四组隔离的受测 agent／教师／动作边界仍以当前配置为准，不因文档合并恢复准入。Claude 已完成的 30 条重审不再列为待办，正式扫描、复审或构造不由本次整理启动。

## 本轮目录收尾

既有授权清理只针对早期 bounded smoke、已被覆盖的通过测试日志及输出目录重复的参数拒绝 console：2026-10-26 01:13 PDT -0700 清理 137 文件、21.89 MiB；2026-10-30 10:44 PDT -0700 清理 685 字节旧通过日志；2026-11-03 23:17 CST +0800 清理 291 字节参数拒绝 console。原始输入、正式盘点与审查、G3、states、邻接结果及原位复核入口均保留，没有借文档收拢清理其他产物。
