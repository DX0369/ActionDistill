# 数据集工具清单项目

更新时间：2026-11-04 14:24 CST +0800。

## 目标与组织

只读审查教师轨迹的工具、动作前历史、候选来源和 Jev 结构准入，不导出训练样本。工具并集不能直接当作固定分类空间；教师轨迹覆盖不等于小执行器实际会走到的状态覆盖。

`src/dataset_tool_inventory/` 实现适配器、共享聚合和 schema，`scripts/` 提供入口，`configs/review_datasets.json` 与 `configs/review_policy.json` 固定格式和准入策略，`tests/` 验证边界。工作区 `data/` 保存原始轨迹及数据卡，项目 `outputs/` 保存独立正式审查和人工证据，`tmp/` 保存有界验证及需保留的原始邻接复核入口。原始数据卡、工具快照和许可证是资产，不属于维护文档整理的删除范围。

当前入口是 `scripts/review_datasets.py`；精确审查方法和操作见 [README](README.md)，当前扫描、分源 G3、验证范围和待解决事项见 [PROJECT_STATE](PROJECT_STATE.md)。旧工具盘点 CLI、registry 及专用适配器已退役，源码保留在原机[本地历史归档](../README.md#历史源码复现)，不以旧清单替代当前审查。

## 环境与运行

使用 `llm-efficiency` 环境及 pyarrow、ijson、transformers、tokenizers。长度估算只读取 `/data/wangsq/model/Mapika-decider-0.8b` tokenizer，不加载模型或联网；审查 JSON 的估算长度不等于最终训练 prompt 长度。

正式扫描由用户启动，必须使用独立新输出目录；smoke 使用独立 `tmp/`。退出码 0、进程结束、结构准入、G3 通过和全量覆盖分别判断。不得改写原始输入、正式审查、G3 明细、已有样本或历史结果；整理文档不授权扫描、复审、样本构造或训练。

## 语义与边界

- 教师只使用配置中的已确认精确白名单；Qwen3.6/3.8 按总参数至少 28B，允许官方 FP8，其他量化不自动纳入。已确认白名单按用户决定视为满足发布时间要求，不用 HF 仓库时间替代模型发布时间，不用目录名、judge 型号或推断别名补教师身份。
- 只从明确原始字段提取动作、任务身份和结果；task_id 缺失保持缺失，outcome 无明确字段则为 unknown。保留失败轨迹，单个工具错误或 completed 状态不自动等同整条任务 failure 或 success。
- 只监督实际观测的下一动作，不伪造完整工具排名；无工具调用、教师不确定或建议器失败不自动产生 ABSTAIN。公开轨迹不提供重新请教师判断才可能获得的弃权信号。多工具集合按当前适配器保留，不任意拍成一个顺序标签。
- 输入必须止于当前动作发生之前；当前动作参数、后续返回和最终答案不得进入该步输入。保留工具名称、定义、候选顺序和权限的来源，不通过删候选、补答案或外部 schema 制造收益。候选不足、当前标签不在集合等排除条件按现行准入实现执行；工具数不计 RESPOND，少于 2 个工具的状态排除。
- S/N/H/I 来源等级、候选完整性、结构 confirmed/conditional 和 G3 是不同判断。恢复 schema、README snapshot 或 H 工具并集不证明逐请求可见或每步可执行；G3 通过不能提升来源等级或消除条件准入。当前 Codex 连续调用及 N 级恢复口径见 README，不沿用旧原子步骤或旧 S 标签。
- 历史移除教师独立 reasoning/thinking 和 assistant 思考块，保留已确认的显式 think 工具及参数；公开文档、代码或工具日志中的字面 `<think>` 引用按来源区分，不机械删除。
- 污染只作告警，不自动排除。τ² 缺失 domain/task_id 为 unknown，明确非 retail 或不在测试 task_id 中仍为 clear；精确指令命中和仅 ID 疑似重叠分开，零命中不证明无污染。
- 准入统计唯一读取 `review_summary.json`；基础计数读取 `summary.json`。隔离、未扫描或解析失败不能记为零；partial 只报告成功解析部分的观测计数。G3 后审裁决优先于旧摘要，正式状态及原人工判定不回写。

## 来源与许可

下表保留既有本地数据卡审查的许可声明和未决条件，不是新增授权。结构准入、教师白名单和 G3 都不替代许可核查；未声明或 `other` 的来源在确认适用条款前不能直接投入训练。数据卡原文和上游条款继续保留原位。

| 来源 | 已记录的许可声明／限制 | 使用边界 |
|---|---|---|
| DeepSeek-v4-Pro-Agent | 数据卡没有 license 字段 | 训练授权待确认 |
| FinancialAuditBench-Trajectories | 轨迹数据卡没有 license 字段 | 任务仓库 FinancialAuditBench 的 MIT 不能推及轨迹训练许可 |
| GPT-5.5-Codex | 数据卡没有 license 字段 | 训练授权待确认 |
| unified-agent-trajectories | 顶层声明 `other` | WildClawBench 及相关模型／工具上游条款待审计 |
| finding-the-right-fit | README 明确禁止训练、微调和蒸馏 | 仅供分析，当前配置明确排除训练 |
| ToolACE、mcp-universe-trajectories、mle-trajectories | 数据卡 front-matter 声明 apache-2.0 | 声明不替代上游条款与其他准入条件；ToolACE 和 mcp-universe 当前另有明确排除理由 |
| gpt-5.6-sol-coding-and-debugging-traces | 数据卡 front-matter 声明 cc-by-4.0 | 保留原声明及适用条件，不能用 G3 代替许可核查 |
| 其他来源 | 现有审查未形成完整许可结论 | 保持未知，不由格式、下载状态或相邻仓库许可证推定 |

公共 coding 轨迹不能直接宣称为 retail/API 工具选择状态覆盖。学生侧错误、重复、空结果或参数冲突的重新采集属于另行定义并授权的研究工作；旧报告中的采集规模和在线对照建议不构成当前协议。
