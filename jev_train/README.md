# Jev 样本构造与 decider 训练

更新时间：2026-11-04 18:11 CST +0800。

当前v2选样、训练、评测与服务入口见本文；研究目标和资产约定见[PROJECT](PROJECT.md)，当前输入、完成状态、结论和下一步见[PROJECT_STATE](PROJECT_STATE.md)。新服务器下载冻结池、选样、环境、训练及校准统一使用[迁移说明](../migration/README.md)。完整冻结池已在本机导出并验收，发布和目标机验收仍待完成。

本文命令供协议明确后由用户启动，每次使用独立输出。已完成批次的完整旧命令、46个样本卡片及旧综述保留在原机[本地历史归档](../README.md#历史源码复现)，公开克隆只提供现行入口。

## 当前 v2：来源配额、共享任务计数与回复优先

`scripts/preview_task_reply_v2.py` 沿用冻结旧 capped 输入、C1、去重和获准领域标签。`configs/selection/task_reply_v2.json` 保留原8k/38k配置；补充的 `configs/selection/task_reply_v2_76k.json` 预览 **8k、38k、76k**，其中前两档用于回归与规模交集核对。选择规则不变：

1. 领域权重仍为 `0.5 × R00领域状态占比 + 0.5/7`；域内恢复 `sqrt(R00来源×领域状态数)` 配额。参考固定为原R00的8,000状态，零参考单元不引入；按正参考单元容量封顶，余额按原权重分配。
2. 来源按digest轮转到固定配额；来源内部每次选累计取样最少的可用任务，同一canonical任务跨来源共享计数。来源配额优先，因此不保证任务全局等额。
3. 任务内合并为保留的 `(model, trajectory_id)` 单元轮转。去重已保证每来源/任务/教师只保留一条轨迹，不因物理trajectory_id相同而合并不同教师。
4. R改为轮界和后继可核实、无终止冲突、无明确框架强制收尾或匹配的ZClaw框架通知边界的规范化RESPOND；来源unknown仍记unknown，但不单因缺少独立来源证据排除R。轨迹内仍按剩余量 `2r/(2r+o)` 取R，R/O各自独立相位spread，无放回。

不再使用C3的25%优先预算，C4关闭；不另设short_turn、direct、严格短停或RESPOND目标配额，训练损失不加权。全局RESPOND **8%–20%**仅为审阅门槛，失败不自动调权或补样。位置箱按保留教师/轨迹单元计算；`trajectories`沿用物理轨迹计数用于历史对照，`retained_units`另报实际轮转单元。

用户确认的两项窄口径继续适用：Hermes非耗尽催促只作诊断、保持R资格；ZClaw当前或后继边界命中完整后台完成通知或runtime-generated内部完成通知时只进O，不扩到单独Subagent Context或sidechain。完整首尾匹配、原始定位和裁决保存在审阅summary，计数及已完成状态见[PROJECT_STATE](PROJECT_STATE.md#当前-v238k训练样本已写出尚未训练)。

三档CPU预览与38k写出的结果、组成比较及训练准备状态统一见[PROJECT_STATE](PROJECT_STATE.md#当前-v238k训练样本已写出尚未训练)。当前数字源为`outputs/curation_preview_task_reply_v2_76k_20261103_205916/summary.json`；下列命令仅供独立复现。

```bash
cd /data/wangsq/code/ActionDistill/experiments/jev_train
v2_review=outputs/curation_preview_task_reply_v2_20261103_201241/summary.json
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 setsid --wait \
  /data/wangsq/software/miniconda3/envs/mapika/bin/python -B \
  scripts/preview_task_reply_v2.py --config configs/selection/task_reply_v2_76k.json \
  --review "$v2_review" \
  --output "outputs/curation_preview_task_reply_v2_76k_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
```

有界真实验证在同一入口加`--smoke`，输出必须改到独立`tmp/task_reply_v2_smoke_<新标识>`；不以smoke分布代替正式配额。

预览仅写summary/report及两个metadata；180条有界原文证据与用户裁决已嵌入summary。`--review`要求固定审阅结构且无未决问题，复现使用上面的正式summary；初次metadata所记临时审阅文件已清理。入口只读复用v1完整的9,712条来源恢复，具名允许其“全部选择完成、仅76k RESPOND门槛失败”的状态；其他未完成或损坏输入显式失败。原文与规范化消息按精确映射核对，两个受影响来源重新读取；本入口不写训练样本或启动训练。工程验收见[状态](PROJECT_STATE.md#当前-v238k训练样本已写出尚未训练)。

## v2 38k 样本写出（CPU）

已交付38k的完成状态、逐条读回及代表性编码验收见[PROJECT_STATE](PROJECT_STATE.md#当前-v238k训练样本已写出尚未训练)；下述writer复现使用新目录。

`scripts/write_task_reply_samples.py` 只写已审阅的v2 38k：复用冻结输入、已保存的轮界审查及原选择函数，核对有序ID摘要、来源/领域配额、R资格和完整统计后，复用C1的`materialize/write_samples`。不重标、不重渲染、不增加过滤或重采样规则；来源未知仍保持未知。逐条读回比较原输入和C1后标签，实际token/候选排列/软标签只对代表样本抽查，不能称全量tokenize验证。

输出仅有`train.jsonl`、`train_ids.txt`、`metadata/{command.txt,run_status.json,curation.json}`。JSONL按原文件offset排列，ID文件保留预览选择顺序。以后训练必须传`--train_ids <样本目录>/train_ids.txt --train_limit 0`，由原`load_train_ids`按seed排序；另传`--validation_samples outputs/samples_eval_normalized_20261025_225810`，不在新训练目录复制评测集。原有short_turn/direct标记保留但不启用C3；R/O与领域标签不写入模型输入。

```bash
cd /data/wangsq/code/ActionDistill/experiments/jev_train
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 setsid --wait \
  /data/wangsq/software/miniconda3/envs/mapika/bin/python -B \
  scripts/write_task_reply_samples.py \
  --preview outputs/curation_preview_task_reply_v2_76k_20261103_205916 \
  --output "outputs/samples_task_reply_v2_38k_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
```

写出smoke使用同一入口加`--smoke`并改为独立`tmp/task_reply_v2_writer_smoke_<新标识>`，从同一38k选择取代表子集。

## 前序方案与已完成队列

早期A从capped按来源分层；B从pool按过滤后任务数sqrt分来源（每轨迹≤8、来源0.5%–15%、主导单工具≤40%、failure≤30%），C再限repeat≤30%，D再删构造loop并在同来源同任务组按success/unknown/failure、决策数、digest取前ceil(n/2)；E用A来源配额加C限制。pool本身每轨迹spread≤16、每来源≤15000，不套capped上限；选择阶段独立随机相位。旧种子对照同时改变抽样、排列和监控，不是只改优化随机性；完整算法/命令保留在本地历史归档，不把这些预算套当前v2。

R11-cap、v1、B2-38k与R四组均为前序批次，不重复挂队列；结果和失败边界见[PROJECT_STATE](PROJECT_STATE.md)。v1的完整回复恢复仍是v2输入，不能因总状态failed删除。历史部署、wave、标注API和旧dedupdom完整命令见[本地历史归档](../README.md#历史源码复现)。

## 冻结输入与领域标签

当前v2冻结旧capped、C1、任务分组、去重和获准领域标签，不重新生成候选或调用标注API。C1后去重键为`(dataset, task_group, model)`，只在有合格状态的轨迹中保留最小`digest(20261015, trajectory_id)`，保留该轨迹在旧capped中的全部合格状态；不补回旧capped之外状态，不把同文件的不同教师合并。canonical恢复来自adapter→build.light正向重建，历史request_hash未持久化，不能称独立比对历史hash。

领域封闭七类为`repo_swe / app_build / data_file / web_research / office_assistant / ml_experiment / other`，按主要交付物定义，精确规则在`src/jev_train/domains.py`。Open-SWE/swe-rebench固定repo_swe、MLE固定ml_experiment、finance固定data_file，冲突失败；OOD不加入训练或领域标注。普通准入要求一致率≥90%、训练other≤10%；本批85/100由用户明确接受，原failed、15处分歧和接受范围保留，其他批次不自动获准。当前canonical_states、annotation_requests、标签与100条审阅均保留，位置见[当前资产](PROJECT_STATE.md#当前资产与依赖)。

### C1、B1/B2与R组的精确定义

| 名称 | 实际规则与边界 |
|---|---|
| D1 / D2 | D1是RESPOND后立即接assistant工具调用；D2只针对RESPOND当前消息显式error/length。缺失终止信息单列，不用未来终止或轨迹outcome替代。 |
| C1 / B1 | D1改后继工具标签；候选缺失、后接另一正文、D2、不可判等按规定丢弃。排除原因可重叠；不扩候选，改gold同步已有action_kind/transition/loop_state，不补造旧训练缺失字段。 |
| C2/C3 / B2 | C1后整轮1–3次调用且以RESPOND结束标short_turn，零调用且修正gold为RESPOND标direct；null调用数进普通池。逐来源先选≤`round(.25*k)`优先状态，Python ties-to-even，余量只从同来源普通池补，不足失败；不固定1039条RESPOND。 |
| 来源配额 | 8k来自B0的`load_selected(index,8000,20261015)`，38k来自A-38k同法选择；B1按digest取前k，B2按C3处理。纯去重R00将deepseek-0813配额552改为实际28，缺口524按其余20来源原权重最大余数一次分配，并列来源名；之后容量不足不再分配。 |
| R00/R10/R01/R11 | 四组8k、单seed20261015、C2+C3开/C4关。以R00实选来源×领域计数`n_ds`为参考：R00/R01领域配额保留原计数，R10/R11七领域近等额；R00/R10域内权重`n_ds`，R01/R11为`sqrt(n_ds)`。零参考单元不引入，缺口只在同领域按原正权重重分配，不复制或跨域补额。 |
| R组C3 | 预算仍是每来源跨领域共享`round(.25*K_s)`，不是各来源×领域单独25%。先保留普通池不足必需的优先前缀，再全来源digest补优先，最后各单元只用普通池补齐；无法满足预算、8000唯一ID或RESPOND 8%–20%则停止。 |

C4已撤销，strict_loop只用于分析，B3/C5不做。旧dedupdom按候选任务数开平方分领域，与R组按R00实选状态计数分配不同；旧规则及预注册判据保留在本地历史归档，不用于当前选型。当前v2已取消C3，不能把上表25%预算套回v2。

### 便携池与默认配方

`scripts/export_selection_pool.py`物化完整C1后去重池150,234状态；正文与annotations分离，包含105个零参考单元状态，默认实际可选的正参考单元为150,129。`scripts/select_pool.py`只读包内数据，不读旧路径或原始轨迹。配方必须恰含`recipe/count/seed/domain_uniform_mix/source_weight_power/reply_priority_weight`，默认`task_reply_v2/38000/20261015/0.5/0.5/2`；回复权重为正整数，其余范围由入口验证。默认值保留原整数分配、sqrt与randrange实现。

去重seed固定20261015，选样seed控制池内选择，训练seed另控制顺序和候选排列。8k/38k/76k各自整数分配，不保证嵌套；任何新配方用新配置和新输出，不自动调阈值。首次转移加`--verify-files`核对全部正文和eval文件；普通选择核对注释/参考内容及正文大小。正式选择RESPOND越界则写摘要、failed且不写ID；smoke限tmp且不能用于正式训练。完整导出、选择、包格式及目标机命令见[迁移说明](../migration/README.md)。

## 本地 Jev 建议服务

`scripts/serve_jev.py --model <最终model目录> --temperature <配套temperature.json> --port <授权端口>`在mapika中只绑定127.0.0.1，bf16/eval/inference_mode、use_cache关闭；模型/温度必填，不生成参数或正文。健康检查须核对实际model、temperature_file和值，旧端口不代表服务仍在运行。

只需官方TypeSafe渲染时，用`--render-only --tokenizer <固定本地Mapika tokenizer目录> --port <授权端口>`；只加载tokenizer，health标`service_mode=render_only`及文件身份，提供`/render`、拒绝`/advise`，不要求自训权重/温度/GPU。新机部署命令见[迁移说明](../migration/README.md)。

`POST /advise`必填messages、tools、`respond: true`。messages使用OpenAI角色/正文/tool_calls，tools为有序平面声明、name必填、description/parameters可选；N/H缺字段不补造。历史arguments必需，字符串原样保留、允许公开custom文本。返回逐位置probabilities(name/probability)、top1、n_prompt_tokens、truncated/messages_dropped/截断计数及温度/模型身份。服务不按名称聚合或做阈值弃权。

同名候选扩展通过`allow_duplicate_candidates=True`保留各声明与位置；训练默认仍拒重名，RESPOND保留名及含其255位置上限不变，推理占位gold不进入prompt。HTTP top1仍是位置argmax对应名称；`jev_bench`记录raw_probabilities/raw_top1后才按名称求和。health/advise均声明`duplicate_candidates="positional"`，客户端逐次验证，旧进程缺字段拒绝。该扩展只经CPU验证，未部署或真实模型smoke。

渲染与下节训练同源；移除独立reasoning/thinking与assistant思考块，保留公开工具输出引用及显式think调用。在线工具顺序不变、RESPOND末尾，不用训练seed/id重排。非法/空候选、>255或保护前缀仍超长返回422，非有限logits/模型失败返回500；服务无fallback，仅`jev_bench`客户端对精确over_length_after_drop使用获准中性消息。`POST /render`走同一渲染/截断路径，另返完整prompt和token IDs，不前向。

## 样本阅读与风险

旧构造版本a9e551e的案例阅读展示23来源各2例，来自各来源文件顺序最早20条：A取第1条，B依次优先动作类型/task_group/outcome/gold不同，平局取最早。它偏向动作多样性，受文件顺序影响，不是随机质量评估或G3重审。完整46卡片及文件/行号/offset/id见[本地历史归档](../README.md#历史源码复现)。

- `model`是逐条教师字段，不由来源名推断；所读gpt-5.5-agent两例实际为gpt-5.4，不外推整个来源。`outcome`是轨迹结果，不是本步正确性。
- S/N/H/I是候选来源等级，分别来自结构化声明、恢复的registry/snapshot、harness/领域重建、此前调用；不是质量分数或无泄漏证明。示例元字段反映当时构造，原池没有I级状态，旧Codex的S标记也不证明请求时工具完整可见。
- 当前gold的参数/正文不在模型输入；过去调用不能当当前动作。TOOL_SET记录同步集合，目标各1/|G|；RESPOND可能是开场说明、澄清或最终回答。训练和下游效果分别验证。
- Toolathlon多工具例可说明集合标签；Kimi-K3/GPT-5.5-Codex例说明“说话后继续工具”；Cursor B已删除114条历史且有可见状态冲突；图像占位或images_replaced=0均不能证明视觉信息完整。failure、unknown、局部测试通过和发布记录结束都不能被改称本任务已完成。

按卡片offset回读一行须核对完整ID；不读取整个大文件：

```python
import json
from pathlib import Path

def read_sample(path, offset, expected_id):
    with Path(path).open('rb') as handle:
        handle.seek(offset)
        row = json.loads(handle.readline())
    assert row['id'] == expected_id
    return row
```

## 输入与构造协议

- 正式审查入口`../dataset_tool_inventory/outputs/jev_review_20261015_005012/`；G3沿旧审查及post_review_adjudication，claude-fable-5采用新重审。build.json明确23个通过/有条件通过来源。exact重复仅删duplicates、不重选keep；污染标记不排除，解析缺口具名计数。
- 先合并同命名空间task_id，再跨来源合并request_hash，MLE例外。初始前缀至第一条非框架assistant：边界跳过与TAU2_GREETING完全相同的assistant，但保留消息。前缀全部user经图片占位、render.text、删完整environment_context、strip，非空片段按原序换行拼接求SHA256；空则missing_initial_request排除，身份不一致报错。skipped_framework_greeting在first_pass按状态累加，不是唯一问候数。含任一OOD成员的组全进test_ood，其余固定seed哈希90/5/5。
- MLE在task_groups.json从`trials/<model>/<competition>/agent/trajectory.json`取competition，task_id=`mle:<competition>`，路径不符报错；按用户确认仅task_id分组，不参与request_hash合并。
- groups报告前20大组、状态p50/p90/max、每来源轨迹/组数/最大组；组比例分母为采样前该来源全部indexed状态。全量仅在来源indexed轨迹≥20且最大组>25%时停止；>5%与组内≥2不同非空canonical task_id仅记录。已逐组认可11组：τ²6组是同开场不同场景，全进OOD；SWE5组是相邻PR同请求，保守合组。旧报告停止结果不重写。missing_initial_request分母=eligible_states−exact_duplicate_states，>1%对有界/全量均停止；大组停止只作用于全量。
- SWE PR规则：Open-SWE用canonical repo校验owner_repo_prN或owner__repo-N，GLM按SWE-rebench解析owner__repo-N；统一`swe_pr:小写owner__repo-PR号`，仅用于分组。未知格式/repo冲突/非法号失败，禁止把PR211的issue210当PR号；Scale-SWE Appendix B、SWE-rebench、PR211及大小写规则来源见task_groups.json。
- capped_mix：failure≤30%（unknown不因失败上限删除）、主导单工具≤40%、每来源≤30,000；确定性采样，来源上限先轨迹轮流取。后续采样/长度排除破坏上限时继续只删超额、不补样，具名计数；极端情况下可剩0。val/test_id/test_ood每来源≤2,000，按状态hash，不套训练配比或改变组归属。harness_distributions在最终配比/长度后统计TOOL/TOOL_SET/RESPOND及S/N/H/I，缺harness保留null。
- 候选原序加RESPOND，prompt.build再打乱并映射255维软标签；N保留描述，只省缺失参数列表；H仅名称，描述≤300字符。复用truncate_history，工具输出/历史参数首尾各2048 token；图片先占位，tool list先把文本/tool_reference渲染为字符串再截断。具名例外tool_reference完整JSON文字化为[tool_reference]，tool_name必填；其他非文本/图片块报错，缺历史参数具名排除。
- state_first精确编码预算32768−64=32704；保护全部system与完整初始前缀（含问候后请求），无非框架assistant时整段保护。省略标记置于完整前缀后，二分找最少删除的最早完整中间轮次，每次及返回前精确编码。仍超长为over_length_after_drop，工具加RESPOND>255为too_many_options，任一来源长度排除>1%停止。训练选项重排后再次检查≤32768，禁止静默截断。

## 环境和测试

数据构造、CPU分析、训练/服务/评测统一使用`mapika`（train-runner角色）。根目录运行，复用只读审查适配器、eligibility和固定`third_party/decider@d6fc522`；环境editable已指向该checkout，启动校验实际导入路径：

```bash
export PYTHONPATH="$PWD/third_party/decider:$PWD/experiments/jev_train/src:$PWD/experiments/dataset_tool_inventory/src"
python -B -m unittest discover -s experiments/jev_train/tests
```

测试覆盖分组/OOD、PR映射、配比、整轮删除、必填字段、候选/gold、soft CE/Brier、排列及长度边界。

源机glibc2.31不能直接用要求GLIBC_2.32的causal-conv1d 1.7.0官方wheel；已按同版本官方源码构建，实际使用系统Ninja。目标机安装/导入/真实GPU验收见[迁移说明](../migration/README.md)，不据源机导入推定A100兼容。旧bf16边界样本曾因算子实现改变argmax，数值限制见[状态](PROJECT_STATE.md#实验历程与验证边界)。

## 样本构造复现

正式样本已完成；以下供用户另行授权复现，每次新目录：

```bash
conda activate mapika
cd /data/wangsq/code/ActionDistill
JEV_SAMPLES_NAME="samples_$(date +%Y%m%d_%H%M%S)"
JEV_SAMPLES_DIR="$PWD/experiments/jev_train/outputs/$JEV_SAMPLES_NAME"
mkdir -p experiments/jev_train/outputs experiments/jev_train/tmp
TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
setsid --wait python -B -u experiments/jev_train/scripts/build_samples.py \
  --review-run experiments/dataset_tool_inventory/outputs/jev_review_20261015_005012 \
  --config experiments/jev_train/configs/build.json \
  --workers 16 --output "$JEV_SAMPLES_DIR" \
  > "experiments/jev_train/tmp/$JEV_SAMPLES_NAME.console.log" 2>&1
```

两遍扫描在项目tmp保存SQLite轻量索引，第二遍只渲染选中状态；按来源/分片并行后严格核对selected覆盖与不重叠，只有全部split提交成功才清理中间分片。render_shards配置及长度/配比检查以完整来源为单位；失败保留可定位诊断。

有界构造用`--limit-trajectories 20 --representative-states --workers 16`，输出限tmp。前20是适配器输入记录而非前20条准入轨迹，第二遍每轨迹取已选首中末≤3条；状态partial，不证明全量或加速。正式构造禁用两个抽样参数；全量第一遍分组审阅用`--group-only --output <项目tmp新目录>`，不与有界抽样共用、不写训练集。最终验收核对build_report准入、重复、划分、配比、长度排除、数量和token；源码和旧验收完整记录见固定快照。

## 训练、索引、指标与恢复

DecisionModel使用Mapika-decider底座全参数微调，在state_first的`Answer: (`处读取原LM head候选字母token logits。单工具/RESPOND为one-hot，多个同步gold为均匀1/|G|，soft CE=`−Σ y_j log p_j`，不是独立sigmoid多标签分类。当前已运行配方Brier权重与label smoothing均0；不直接优化参数、完整任务成功、输出token或门控后行为。AdamW、weight_decay0、gradient clip1、activation checkpointing开启、max_tokens/max_len32768；微批按token动态组成，同状态数/epoch不等于等token/等步数或全局样本等权。

已完成run的checkpoint保留步数见[状态文档的保留清单](PROJECT_STATE.md#已保留的训练检查点)。最终`model/`仍是正式评测输入；已删除的中间步不能用于恢复，留存规则不自动应用于运行中的训练。

SampleIndex共用于训练、监控、评测和分析。无缓存时流扫JSONL，验证必填字段、全文件ID唯一及正整数n_prompt_tokens，只存offset/id/dataset/token数；读取/编码时仍要求实际token数相等，不改长度或回退。缓存固定`outputs/index_cache/<样本目录名>/<split>.json`，校验源绝对路径/size/mtime、列数/顺序/ID及固定抽样行，原子提交。它不是全内容hash；不一致显式报缓存路径，确认后删除对应缓存再重建，不自动fallback，不改canonical JSONL。

训练majority仅统计实际选择、单gold且非RESPOND的逐来源工具频数，并列工具名字典序；不改变选样/批次。训练监控数据必须显式指定本批规范化集合，省略validation_samples才使用samples。

默认val监控300条：每来源先1条，余量按容量比例及最大余数，容量不足失败；默认每100步，本轮显式间隔覆盖默认。activation_checkpointing仅on/off、默认on；先grad_ckpt=False构造，再显式use_reentrant=False，单/多卡一致，不按显存切换。训练结束保存model、training_majority_tools和completed，不评测测试集或拟温度，不生成eval.json；completed只指训练完成。

指标：loss=均匀多gold软CE；total_loss=CE+brier_w×Brier用于反传。acc是top1命中targets正值集合；总体/各来源报告soft NLL、Brier、均匀随机与always_respond_acc，并按action_kind/outcome/candidate_level报告acc/count。macro_by_dataset为非空来源acc等权，总体acc按状态加权。15-bin ECE仅单gold，等宽top1置信度箱（末箱含1），按各箱在单gold中的比例加权绝对校准误差；无单gold为null，ece_count记实际分母。

val/monitor/test_id的train_majority_tool_acc仅用实际训练选择、逐来源最常用单gold工具，排除RESPOND/TOOL_SET；test_ood的oracle_majority_tool_acc按本split所有gold工具成员（含TOOL_SET，除RESPOND），只是描述oracle。并列字典序，来源缺统计明确null；定义写baseline_definition，不用于校准。

配置JSON拒绝布尔值（含嵌套），smoke独立CLI；抽样也不跳过全文件ID唯一检查。logits含NaN/+inf或整行无有限值失败，-inf只作无效候选mask。学习率按step+1调度，warmup首步非0。独立评测仅用所选val soft NLL在[0.3,5]的120点log grid拟单温度，端点标at_grid_boundary；据此算ID/OOD，唯一最终源eval.json，温度另存temperature.json，不写训练run/decider_config，部署显式配套。

checkpoint在完整累积后保存model、optimizer、step、全局epoch/batch游标、loss历史、按rank的Python/NumPy/Torch CPU/当前CUDA RNG（rank_rng）。恢复必须同out、run_name、模型/数据、超参，只加`--resume <out>/checkpoints/step_XXXXXXX`，配置不同失败；SwanLab按run_status原run id以resume=must继续，恢复指标按resume_N/区分；原日志只追加明确恢复边界，验收重建有效步序列，原失败分支不删除。smoke须`--smoke --swanlab_mode disabled`；训练不接受eval_limit，新配置不可直接续旧checkpoint。

## 双卡训练

`accum`为每卡次数：双卡accum1与单卡accum2使用相同全局微批序列、微批均值权重、步数、学习率和裁剪，不按全局样本数重新加权，尾微批仍用批内均值；双卡默认accum2会改成每步四微批。相同配方和执行/恢复通过不等于已证明数值等价。

无WORLD_SIZE/RANK/LOCAL_RANK走单卡，不初始化组；torchrun三变量须完整、可见卡数=world_size。NCCL超时2小时，broadcast_buffers=False、gradient_as_bucket_view=True、find_unused_parameters=False，非末本地微批在no_sync中前反向；unused parameter错误不跳过。全局每步推进world_size×accum微批；epoch末尾按现有顺序接下一epoch以完成整步，不丢尾微批，也不复制选样ID。

仅rank0写盘/SwanLab/majority/checkpoint/固定val；先存再评估，其他rank等barrier，评估用未包装模型；各rank更新同optimizer并各自恢复rank_rng。配置/SwanLab记录world_size，恢复卡数必须相同，否则显式报checkpoint/current值。无world_size/rank_rng的旧checkpoint只能用原commit续，不转换。

日志loss/total_loss为跨rank微批等权均值；tokens为全局真实prompt token（无padding）。step_seconds包含读/编码、前反向、同步、optimizer，排除保存/监控/导出；seconds为训练循环累计墙钟。max_memory_allocated取rank最大，completed另存by_rank。rank0将SIGTERM记录failed，退出销毁进程组。

以下是源机2B×38k模板；模型、样本、IDs、验证集、run name和两卡必须明确，输出为新目录。`train_pilot.json`默认0.8B/2000条、eval/save100步，本模板显式覆盖为当前38k配方。目标机使用[迁移说明](../migration/README.md)的解释器和路径，不将双卡直接改八卡。

```bash
conda activate mapika
cd /data/wangsq/code/ActionDistill
export PYTHONPATH="$PWD/third_party/decider:$PWD/experiments/jev_train/src"
: "${JEV_DDP_GPUS:?填入两张授权空闲卡}"
: "${JEV_SAMPLES_DIR:?填入已确认的训练样本目录}"
: "${JEV_VALIDATION_DIR:?填入规范化验证样本目录}"
: "${JEV_TRAIN_IDS:?填入已确认的训练ID清单}"
: "${JEV_TRAIN_RUN_NAME:?填入本次SwanLab运行名}"
JEV_TRAIN_ARGS=(
  --config experiments/jev_train/configs/train_pilot.json
  --model /data/wangsq/model/Mapika-decider-2b
  --samples "$JEV_SAMPLES_DIR" --validation_samples "$JEV_VALIDATION_DIR"
  --train_ids "$JEV_TRAIN_IDS" --train_limit 0
  --accum 1 --lr 1e-5 --seed 20261015 --warmup 150 --epochs 1.0
)
JEV_SMOKE_OUT="experiments/jev_train/tmp/${JEV_TRAIN_RUN_NAME}_smoke_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$JEV_DDP_GPUS" setsid --wait torchrun \
  --standalone --nproc_per_node=2 experiments/jev_train/scripts/train_jev.py \
  "${JEV_TRAIN_ARGS[@]}" --out "$JEV_SMOKE_OUT" --run_name "${JEV_TRAIN_RUN_NAME}_smoke" \
  --smoke --swanlab_mode disabled --max_steps 4 --save_every 2 --eval_every 2 --monitor_val_size 30
```

真实smoke需核对前反向、有限loss/grad、监控、两rank checkpoint/RNG与导出，诊断结束前保留失败证据；通过不证明训练效果、逐位等价或最长样本显存安全。核对本次长度和显存后，由用户启动正式命令：

```bash
CUDA_VISIBLE_DEVICES="$JEV_DDP_GPUS" setsid --wait torchrun \
  --standalone --nproc_per_node=2 experiments/jev_train/scripts/train_jev.py \
  "${JEV_TRAIN_ARGS[@]}" \
  --out "experiments/jev_train/outputs/runs/$JEV_TRAIN_RUN_NAME" --run_name "$JEV_TRAIN_RUN_NAME" \
  --swanlab_mode cloud --max_steps 0 --eval_every 2000 --save_every 2000 --monitor_val_size 300
```

历史DDP执行与恢复证据、未定位的GPU5/7 NCCL停滞和数值限制见[状态](PROJECT_STATE.md#实验历程与验证边界)，不把单卡通过替代指定双卡链路验收。

## 样本统计复现（CPU）

正式样本不含option_names，沿用tool_name()解析，不回填。脚本流式读行，只留ID集合、频数及有限编码缓存；默认新建sample_stats目录，输出summary与两个metadata，smoke指定项目tmp：

```bash
conda activate mapika
TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
setsid --wait python -B -u experiments/jev_train/scripts/sample_stats.py \
  --samples experiments/jev_train/outputs/samples_20261015_175031
```

按split/来源/来源×split报告选项token数及prompt占比，nearest-rank p50/p90/max。按canonical选项序计标签、换行、工具文本，排除Question/Options标题和Answer；>10项复用上游分段编码，≤10项用联合编码offset，跨边界token算选项；分母是存储n_prompt_tokens，不是训练重排后长度。

无原始消息边界，无法精确算“仅剩初始前缀”；具名代理prefix_only_visible_proxy_fraction=messages_dropped>0且context末尾独立行恰为省略标记，分母全状态，仅说明标记后无可见文本，公开引用/后置受保护system可影响它。harness复用并核对build_report，不从来源推断；动作由gold/tool_name解析；ID连带进入OOD是最终保留人群，非配比前总体。

工具声明审计只覆盖14/23来源，其余未覆盖且源有解析失败，不能宣称全量无歧义。未来构造拒绝工具名含`': '`或`' (params:'`，旧样本不回填；旧统计数值按固定快照查阅。

## 独立评测、配对与恢复

mapika评测不接SwanLab。`--run`（含model/）与`--model`二选一，out不得在run/模型/样本内；configuration记model_source，不读decider_config，worker关闭use_cache。majority、seed、三limit必填，limit0为全量。使用协议指定样本和获准`--gpus`，换终端显式恢复路径，不找latest。

旧工具名解析把仅含参数的`write_file (params: path, content)`误取为`write_file (params`；现取`': '`与`' (params:'`最早分界。旧pilot part的名称诊断不迁移；acc/NLL/ECE、训练prompt/targets未受影响。旧sample_stats仅用名称区分RESPOND，其token/动作统计不变。需新诊断时另开评测目录，不能原地改旧结果。

val/test未套训练tool_fraction=.4/failure_fraction=.3，但构造每来源≤2000、评测每split分层1000，因此是“保留集评测子集准确率，未做训练配比重平衡”，不是原始自然分布。majority_tool_candidate_count仅报告训练多数工具在候选出现次数，未出现可合法；无训练统计为null并报available_count。

每split逐来源respond记录gold/predicted/TP及precision=TP/predicted、recall=TP/gold，零分母null；overall只合并RESPOND。tool_prediction按gold频次列前10非RESPOND（多gold逐成员计1，并列名称）；top_confusions列前5错误对，仅top1不在完整gold集合才错误，多gold错误归因名字排序首项。定义写diagnostic_definition，不发SwanLab、不维护“结束类工具”清单。

统一模板；底座把run换model并仍提供匹配训练选择的majority：

```bash
cd /data/wangsq/code/ActionDistill/experiments/jev_train
JEV_RUN='<训练run目录>'
JEV_EVAL_SAMPLES='<本次协议指定的评测样本目录>'
JEV_MAJORITY='<匹配训练选择的training_majority_tools.json>'
JEV_EVAL_GPU='<已批准GPU>'
JEV_EVAL_OUT="outputs/evals/jev_$(date +%Y%m%d_%H%M%S)"
setsid --wait /data/wangsq/software/miniconda3/envs/mapika/bin/python -B -u scripts/eval_jev.py \
  --run "$JEV_RUN" --samples "$JEV_EVAL_SAMPLES" --majority "$JEV_MAJORITY" \
  --out "$JEV_EVAL_OUT" --gpus "$JEV_EVAL_GPU" --seed 20261015 \
  --limit_val 1000 --limit_test_id 1000 --limit_test_ood 1000
```

`--compare_to <参考评测目录>`要求参考completed、相同seed，各split唯一且完全相同ID；按ID核对targets整行和nopts。命中严格targets[i,argmax]>0，不比较tool_names或要求名称唯一；解析差异不妨碍位置配对。报告当前减参考的acc_difference及improved/regressed/discordant；只有eval.json无part的旧内置评测不能配对。新part必须有transition/copy_prev_hit，缺字段失败，旧产物不迁移。

多卡仅用获准GPU；父进程spawn每卡worker并独占状态写入。选中条目round-robin分片，片内按(n_prompt_tokens,id)排，每200条原子保存part，含id/logits/targets/dataset/outcome/candidate_level/nopts/排列后tool_names/transition/copy_prev_hit；parts保留用于恢复/分析，stdout报进度/ETA。

中断复用同命令/out；完成part经ID/形状验证后跳过。seed、limits、分片数、模型/样本/majority或model_source变化须新目录。`--finalize_only`只汇总已保存part，缺part或覆盖不符失败；旧目录缺model_source不能用新版原地resume/finalize，compare_to也要求新字段，不加迁移/fallback。B1/B2只配新B0norm，不套旧pool口径。

`transition`使用完整未截断历史：无前一assistant动作为first，当前gold集合是前动作集合子集为repeat，否则change，RESPOND也算动作。copy_prev_rule固定`single_member_or_multi_equal`：前动作去重后仅一名时属于gold即命中，多名时须与gold集合完全相等，无前动作不命中；repeat与copy命中不是同义。缺prev_action_names/transition显式失败，空切片count0/acc null。

省略`--splits`仍为三split，limit0表示全量而不是跳过。可先`--splits val`校准，再在独立test输出使用`--splits test_id test_ood --calibration_eval <同模型completed val目录>`；部分completed只覆盖声明split，比较默认要求三split，不能取输入交集。模型或温度快照、split变化均须新目录。

唯一最终源eval.json，另存temperature.json、shards及两个metadata。completed训练与completed评测分别验收；离线标签命中不直接证明executor成功率改善。本页命令不构成启动授权。

## 已完成分析复现

下列保留入口读取已有数据/parts；重新生成正式统计仍需授权和新目录。回复边界和原始记录恢复逻辑由主线共享，历史报告入口不作为运行依赖。完整旧命令及抽样验收保留在[本地历史归档](../README.md#历史源码复现)。

| 入口 | 精确口径与前置 |
|---|---|
| `scripts/compare_evals.py` | `--eval LABEL=DIR`可重复，`--reference LABEL --out NEW_DIR`；同seed/ID/targets/nopts严格配对。按task_group整体bootstrap10000次、seed20261015、排序秩249/9749，另报任务等权差/符号检验，无改善退化时p=1；`--domain-labels DIR`才增领域切片，去0813不重新拟温。 |
| `scripts/analyze_short_turn.py` | `--samples DIR --semantics JSONL --policy-impact DIR --eval LABEL=DIR --output NEW_DIR`；显式提供既有回复分类及终止证据，先`--audit-only`关联后继特征，未关联保留unknown、任一split>5%未关联停止。每模型使用自己的温度；非空组零不一致McNemar p=1，空组p=null，n<30仅参考。 |
| `scripts/analyze_respond.py` | `--evaluation DIR --training-run DIR --output NEW_DIR`；训练选择从原命令恢复，不能用旧8k与pool的2506行交集代表全部训练。原温度固定后在logits/T上给RESPOND加b，仅val soft NLL在[-1,3]步长0.1选b；OOD只作固定敏感性检查。零gold分母recall=null。 |
| `scripts/analyze_teacher_loops.py` | `--training-analysis DIR --pool DIR --evaluation DIR --output NEW_DIR`；只从原可见历史恢复并核对ID/教师/候选/gold，当前gold不参与循环判定，不读取OOD预测；缺轮界或片段历史为unknown。 |
| `scripts/sample_stats.py` | 见上节；只读既有样本，描述统计不替代质量评估或正式在线成功率。 |

### 停止、位置与循环的不同定义

严格短停止只在当前规范化gold=RESPOND时定义：后继新用户、本轮此前1–3次调用，排除明确error/length及终止冲突；missing/null终止元数据仍保留并单列。short_turn分析分类依次为后继/调用数未知→unknown，后接工具→mid_commentary，零调用→no_prior_call，≥4次且新用户/轨迹末尾→long_turn_or_end，1–3次EOF→short_end，1–3次新用户且无异常→short_turn。EOF不是任务完成真值，整轮训练short_turn可含工具决策，二者不可互替。

原pool的trajectory_position按完整indexed状态的`(turn_index,id)`排序后从0编号，分母trajectory_decisions；first表示无前动作，不等于轨迹首状态。当前v2位置箱的分母是保留的C1合格教师/轨迹单元状态数，另报原消息位置0，不冒充原轨迹总长度。

构造loop_state要求末三条assistant各单调用、同名同参且等于当前单工具gold、期间无user；教师循环分析在canonical用户轮内按原并行列表展开最近三次同名同参调用，**不含当前gold**；线上loop_guard则看最近两次实际调用及当前Jev top1，属于另一协议。function空参数字符串具名视{}，损坏非空JSON按原文比较；Codex custom按已定位原调用类型比较input原文，不以解析失败自动切换语义。结构循环含正常轮询/分页及重试，不自动认定有害。

## 数据机制交叉审计

`scripts/analyze_training_mechanism.py`按`configs/analysis_training_mechanism.json`只读B2-8k、R00、R01、R11及B2-38k的实际选样、原始教师轨迹和训练日志；不改数据、不推理、不训练。环境使用已含Parquet读取依赖的`mapika`，显式禁用CUDA。

- 来源、教师、领域分别用`dataset`、`model`和既有任务组领域标签。规范化gold以实际train.jsonl为准，逐ID核对C1恢复表；保留/新增/移除按配置中五对ID集合报告。
- 严格短停复用`analyze_short_turn.classify`：仅当前规范化RESPOND、后继新用户、本轮此前调用1–3次；明确error/length及终止冲突排除，missing/null终止信息保留并单列。训练`short_turn`标签另报，不能当停止正例数。
- 报错回查本用户轮原始tool返回：只认布尔`is_error/isError=true`或非空结构化`error`值，null/false/空值不判错；其他值类型记不可判。不猜自由文本或exit code。原始标志由tool call ID关联，缺返回、无法关联或历史不完整记unknown；零调用单列。已见显式错误优先，仍保留其他返回缺失原因。“未见显式错误”不等于成功。
- 损失系数按原`batches_by_tokens`重放，包括整步补齐时跨到下一epoch的样本；每次出现的系数为`1/(world_size * accum * microbatch_size)`。逐更新token总数须与有效训练日志一致；恢复后被覆盖的旧尝试不重复计入。各组系数总和等于更新数，除以更新数得到份额。该量不含学习率、梯度范数或Adam动力学，不称为真实梯度贡献。
- `summary.json`保存交叉表与集合变化，`states.jsonl`保存紧凑逐ID分类/证据位置/各组系数，`report.md`只作机制描述；不根据相关性归因在线成功率，也不自动改变配方。

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /data/wangsq/software/miniconda3/envs/mapika/bin/python -B -u \
  experiments/jev_train/scripts/analyze_training_mechanism.py \
  --out "experiments/jev_train/outputs/analysis_training_mechanism_<新标识>"
```

工程验收：10项CPU针对性测试通过；153个跨来源/动作/调用位置状态的真实smoke通过，五组完整训练的每步token重放一致。初次mapika环境缺少pyarrow，改用项目已有的源数据读取环境解决，未新增依赖或改变分类。

## 历史协议索引

已结束训练/评测/wave、R组恢复、R11-cap/v1、旧dedupdom及46样本卡片保留在原机[本地历史归档](../README.md#历史源码复现)。原metadata/command/configuration与正式结果仍是批次事实来源，不能靠文档表格覆盖历史分母、温度或原失败。
