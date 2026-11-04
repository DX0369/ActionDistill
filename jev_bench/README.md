# Jev benchmark

更新时间：2026-11-04 12:19 CST +0800。

本项目统一接入BFCL、τ²、ToolHop、API-Bank、ToolSandbox与StableToolBench。目标/固定资产见[PROJECT](PROJECT.md)，CPU和真实联调覆盖、原失败及待办见[PROJECT_STATE](PROJECT_STATE.md)。下载、环境重建、服务命令、目标机部署及启动顺序统一见[迁移说明](../migration/README.md)；本文维护运行、方法、统计和恢复合同。当前正式主表未启动，最终Ours和全量PA映射尚待确定。

## 先做离线预览与测试

默认`run.py --config <配置>`只预览，不调用模型/服务、不需GPU。下表保留六项独立运行/评分环境；新机候选环境合并须先验收，不能自行把本机独立评分环境替换掉。

| benchmark | 本机解释器环境 | 预览配置（configs/） | CPU回归目录 |
| --- | --- | --- | --- |
| BFCL | trust | bfcl_mt_native_smoke.json（4类各2题，共8条） | tests |
| τ² | llm-efficiency | tau2_native_smoke.json（三域各1，共3条） | tests_tau2 |
| ToolHop | jev-toolhop | toolhop_cpu_preview.json（0/44两题） | tests_toolhop |
| API-Bank | jev-apibank预览/评分；trust生成 | apibank_cpu_preview.json（六类各1点） | tests_apibank |
| ToolSandbox | jev-toolsandbox | toolsandbox_cpu_preview.json（两个本地场景） | tests_toolsandbox |
| StableToolBench | jev-stabletoolbench | stabletoolbench_cpu_preview.json（六组各1题） | tests_stabletoolbench |

例如从本项目目录运行BFCL；其他项按表替换解释器、配置和测试目录：

```bash
cd /data/wangsq/code/ActionDistill/experiments/jev_bench
JEV_BENCH_PYTHON=/data/wangsq/software/miniconda3/envs/trust/bin/python
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 LITELLM_LOCAL_MODEL_COST_MAP=True \
  "$JEV_BENCH_PYTHON" -B scripts/run.py --config configs/bfcl_mt_native_smoke.json

JEV_APIBANK_SCORING_PYTHON=/data/wangsq/software/miniconda3/envs/jev-apibank/bin/python \
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  "$JEV_BENCH_PYTHON" -B -m unittest discover -s tests -t . -v
```

τ²预览/测试设置`LITELLM_LOCAL_MODEL_COST_MAP=True`；API-Bank离线预览设置`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`并使用已到位MiniLM。共享HTTP fixture用loopback假服务及SDK mock，需允许本机临时端口，不做外部推理。PA命名回归在llm-efficiency下用`PYTHONPATH=src:. python -B -m unittest discover -s tests_pa_tool -t . -v`。这些脚本actor、CPU预览和import检查不能当模型准确率或真实smoke；测试临时文件只在本项目tmp创建和清理。

## 扩展正式配置与统计

正式全量固定BFCL四类800题、τ² base278题、ToolHop995题、API-Bank1307点（六类分别报告）、ToolSandbox1032场景（129家族×8变体）、Stable六组765题；每题每条件一次，不用smoke子集代替。四项扩展显式`mode=formal`才允许outputs，强制全量；cpu_only/smoke限tmp、无推断比较。BFCL/τ²保留原合同，不加mode。API-Bank计划冻结canonical source_id，ToolSandbox冻结family_id/variant/categories，恢复按计划核对，不从复合ID猜来源。

完整主表为Native、官方Jev、Ours、ActionOPD、STAR、PA-Tool，DeepSeek可选；主要比较只显式列Ours对其余已纳入条件，不自动展开两两。当前四份五基线配置为Native、官方Jev、ActionOPD、DeepSeek、STAR，comparisons空；Ours最终权重/温度/策略暂缓，PA等待从零全量映射，不填虚假路径。

| configs/中的阶段性配置 | 环境 | 每条件输入 | 五条件计划 |
| --- | --- | ---: | ---: |
| bfcl_mt_baselines_formal.json | trust | 800题 | 4000 episodes |
| tau2_baselines_formal.json | llm-efficiency | 278题 | 1390 episodes |
| toolhop_baselines_formal.json | jev-toolhop | 995题 | 4975 episodes |
| apibank_baselines_formal.json | trust生成／jev-apibank评分 | 1307点 | 6535决策点 |

配置仍保留本机Qwen8001、ActionOPD8005、STAR8006、本地renderer18767及旧TypeSafe的ours_jev绑定。新机由`migration/deploy.py`生成独立配置、TypeSafe改绑tokenizer-only jev_render；health验证渲染能力，不能当动作建议器。预览已通过，端口/目标机服务尚待验收，正式未运行。

扩展统计统一报告treatment−reference：ToolHop精确任务配对置换；API-Bank按来源对话整簇、ToolSandbox按基础场景家族（8变体同簇）、Stable按query配对且六组分别推断。后三区组交换两条件，双侧、非零簇≤20穷举，否则100000次Monte Carlo、plus-one。95% percentile区间10000次、seed20261015，整簇抽样保留零差簇。API-Bank仍点等权，每次重算簇内差值和/点数和，六类分别推断；L3 Sample Accuracy仅完整245 API点/50对话时描述。ToolSandbox仅总体similarity为主指标，重叠官方类别仅描述。Stable使用每query聚合SoPR，judge重复不算独立样本；目录保持ZIP类别首次出现及类内原序。

每benchmark的全部预声明主要比较×主指标构成一个Holm家族，保留原始p；缺分检验占家族大小、p=null，区间只是逐项95%而非同时区间。缺分不补0、不取已评分交集；受影响类别均值/比较null。ToolHop具名PA接口失败计false的例外保留，其他三项不伪造官方0分；成本含全部attempt、未知usage不补0。BFCL/τ²仍用各自类别/领域权重与原配对口径，新Holm不倒改旧项目。

资产齐备后生成可审查配置，不启动模型/评分、不读key，open('x')防覆盖：

```bash
cd /data/wangsq/code/ActionDistill
JEV_FORMAL_BENCHMARK=toolhop
/data/wangsq/software/miniconda3/envs/trust/bin/python -B \
  experiments/jev_bench/scripts/prepare_formal_config.py \
  --template "experiments/jev_bench/configs/${JEV_FORMAL_BENCHMARK}_main_smoke.json" \
  --conditions native official_jev ours actionopd star pa_tool \
  --ours-model "${JEV_FINAL_MODEL:?请指定最终Ours权重}" \
  --ours-temperature "${JEV_FINAL_TEMPERATURE:?请指定配套温度文件}" \
  --ours-gate "${JEV_FINAL_GATE:?请指定off、G1或G2}" \
  --pa-mapping "${JEV_PA_MAPPING:?请指定该benchmark完整冻结映射}" \
  --output "experiments/jev_bench/configs/${JEV_FORMAL_BENCHMARK}_main_formal.json"
```

benchmark可选bfcl_mt/tau2/toolhop/apibank/toolsandbox/stabletoolbench；可选DeepSeek追加deepseek_advisor。不列ours时不传其三参数，不列pa_tool时不传映射。核对资产、地址、比较并提交配置，再按对应环境预览；PA在任何服务检查前还核验内容/全计划覆盖，文件存在不等于通过。

Ours门控off为prefer，G1/G2原规则；默认top1，显式`--ours-adaptive-top3-threshold 0.5`才启用：先门控，放行且top1<0.5才展示最多三项，否则原top1，不重归一化。prediction/first_action_matches仍指top1，hint_kind/shown另记实际展示；neutral不能称采纳。Stable的真实Finish与虚拟RESPOND保留各自概率/逻辑项，提示二者均对应Finish。官方Jev/DeepSeek不变；这些入口只是CPU工程能力，不预选最终策略。

## 配置约定

benchmark.name、canonical task_ids、顺序seed显式配置。相同executor URL串行，不同URL各一链。τ²记录`seed + 打乱后任务序号×trials + trial`，参与者set_seed为空操作、不向LLM传seed，只固定任务顺序。

executors声明URL、served model、绝对路径、预算、YaRN和采样；advisors.type必填。jev_local在health/advise均核验模型/温度；typesafe声明官方URL/版本、api_key_env、render_advisor；deepseek声明官方URL/移动别名、key、16384输出、high effort。六项所有执行器统一thinking、T=0.6/top_p=0.95/top_k=20/min_p=0、8192输出；PA命名另用独立协议。当前vLLM0.19.1的input_text字符预检与input_tokens超限须连同HTTP400/错误类型/明确上下文消息核验，其他400不改判超限。

比较各方法系统和完整成本，不称等输入、等预算或纯训练效果；任务、官方公开来源、执行/评分规则共用，轨迹可分化。

| 条件 | 可见输入与职责 |
| --- | --- |
| native／actionopd／star | benchmark官方接口；API-Bank为文本，其余为转换tools；独立执行 |
| official_jev／ours | 同一Qwen的软建议器；共用本地公开候选摘要：名称、已有描述前300字符、顶层参数名，不展开参数说明/类型/嵌套；共用历史预处理，不证明远端内部模板/分词/最终输入相同 |
| deepseek_advisor | 当前公开状态的一步建议；通常全公开schema，API-Bank原发布文本；只取首动作名，参数不执行，不推进另一环境 |
| pa_tool | 原Qwen改工具/顶层参数名及规定引用；公开schema/描述和隐藏边界保留，准备成本独立记 |

候选摘要不代表整个请求无参数细节：API-Bank公开接口文本仍在history，ToolHop不自动补schema入history；不恢复ToolSandbox隐藏说明、不修补Stable转换来加信息。更完整Jev候选须另定新方案，本轮不改渲染/训练资产。

advice支持none/neutral/prefer/gate；gate必填rule/threshold/loop_guard：G1=respond_prob_ge/0.1，G2=top1_prob_lt/0.3，loop_guard=true。可只运行一个gate。近期调用按当前用户轮与已解析顺序更新（含并行、无需执行成功）；最近两次调用同名同参且当前top1仍为该工具时触发loop，再检查confidence；top1为RESPOND时照常提示回复。提示只在当前请求副本末尾注入一次，参数由执行器生成；原生FC非流式、tool_choice=auto，API-Bank保持文本。BFCL执行器保留历史reasoning；建议器清除执行器私有推理，不见未来/GT/隐藏状态。门控先调用Jev，不省其推理。comparisons显式列[treatment,reference]。

同名候选使用sum_by_tool_name：保留全部声明和原序、不合并schema。服务按位置给概率/原argmax；保存raw_probabilities/raw_top1，再按原名求和为probabilities/top1、标记aggregation。建议/门控/loop/采纳用聚合结果，并列按名称首次出现。禁止RESPOND工具名、最多255位置；训练构造默认拒重名。health/advise须有duplicate_candidates="positional"，缺失即拒绝。远端位置身份见[下文](#远端建议器)。

## 与旧base批次的区别

BFCL/τ²统一YaRN窗口131072、原始窗口32768、factor4，输出8192；渲染后输入上限122880含工具/system/模板。采样为0.6/0.95、top_k=20、min_p=0。BFCL保留官方循环、评分、Python语法提示及holdout工具原文，不补写语言后缀；本项目chat completions/vLLM解析器与官方QwenFC的completions、预算不同，不能直接对齐榜单。旧base的32k/无后缀结果不得混入。

Jev仍使用既有服务和训练渲染，prompt预算32704。它会截断单段工具输出/参数，并在必要时删除早期历史单元，保护初始前缀。逐步保留`truncated`、`messages_dropped`和`truncation_counts`，按类别/条件汇总已知请求的截断比例，缺测不记零。

## 服务检查与后续执行边界

`--check-services`只检查已部署服务：Qwen `/v1/models`须匹配真实路径和131072上限，Jev `/health`须ready且模型/温度一致。NO_PROXY/no_proxy只影响当前进程；端点检查不证明YaRN/显存峰值可用，须核对参数/日志，不为server_info启开发模式。部署为vLLM0.19.1、hermes/qwen3 parser，读取canonical message.reasoning、不兼容别名；用hf-overrides配置yarn/factor4/original32768、rope_theta1000000，不改共享config.json。

预检不推理：τ²只查模拟用户key非空、remote_checked=false；TypeSafe只查key/renderer身份、不调systemone；DeepSeek仅GET官方models、inference_checked=false。不输出/保存key，localhost绕过不影响远端代理。只有显式--check-services访问端点。

正式/真实smoke均须明确资源、最终资产/映射和授权；入口不启停模型。代表性smoke须检查全部最终条件、长输出/超限、官方评分、身份/提示/截断、失败分类、显存与耗时，parser允许合法空思考块。旧有界联调/native模板不替代完整方法验收。

### 主表接入的CPU准备与后续smoke

六份main_smoke保留Native/官方Jev/Ours/ActionOPD/DeepSeek/STAR，PA独立模板；tmp、一次trial、comparisons空。A-38k/T=1.5331624596487692/G1仅临时链路占位，不是最终Ours。模板端口不是实时资源，按迁移入口生成独立tmp配置。

| 模板前缀 | 六条件计划数 | PA独立任务数 | 额外依赖 |
| --- | ---: | ---: | --- |
| bfcl_mt | 48（四类各2） | 8 | — |
| tau2 | 18（三域各1） | 3 | DeepSeek非思考user |
| toolhop | 12 | 2 | 官方工具依赖 |
| apibank | 36点 | 6 | MiniLM/隔离worker，部分API联网 |
| toolsandbox | 12 | 2 | DeepSeek user，smoke限已选本地场景 |
| stabletoolbench | 36 | 6 | ToolBench key、DeepSeek模拟器/judge |

BFCL/τ²六条件smoke共66条，去DeepSeek时55；正式六个必做条件共6468 episode、含DeepSeek7546。单卡可按executor拆为Qwen四条件/ActionOPD/STAR三个tmp配置，删未用executor/advisor、题集不变、不重跑已完成子配置，不为显存改采样/上下文。

所有项共用预览、身份检查及获准后执行入口，按表选解释器：

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost LITELLM_LOCAL_MODEL_COST_MAP=True \
  "$JEV_BENCH_PYTHON" -B scripts/run.py --config "${JEV_CONFIG:?请指定已核对配置}" --check-services

NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost TZ=Asia/Shanghai \
LITELLM_LOCAL_MODEL_COST_MAP=True setsid --wait "$JEV_BENCH_PYTHON" -B -u \
  scripts/run.py --config "$JEV_CONFIG" --execute --run-id "${JEV_RUN_ID:?请指定新标识}"
```

BFCL/τ²/API-Bank有界smoke可把未选类别设空列表，仅tmp/一次/无comparisons，canonical类别键完整且至少一类非空，不变正式全量。PA模板先替换REPLACE_ME映射，不建advisor、不需Jev/TypeSafe key，模拟用户/Stable后端凭据仍需。密钥从环境读取、不写JSON或命令。旧native/四条件模板保留，同入口操作，不再内嵌重复端口配置生成器。

### STAR-4b单条件准备

冻结权重/tokenizer/template、路径/revision见PROJECT。原config40960未启YaRN，tokenizer标131072不证明扩窗或长窗口质量；索引total_parameters=196096为作者计数异常，真实header4,022,468,096、原文件保留。统一thinking/采样/8192/131072不是作者长上下文验证。

服务用[迁移入口](../migration/README.md#服务部署配置和建议策略)的--kind star显式指定资源，不重复下载。STAR-only tmp只保留star executor/condition、移除advisors；首次BFCL每类第一题共4，τ²三域各1共3，其余沿模板。无需Jev/TypeSafe；τ²/ToolSandbox仍需DeepSeek user，Stable需ToolBench和DeepSeek模拟器/judge，正式ToolSandbox需RapidAPI。核对star-4b/路径/131072，用既有episode/metrics/metadata格式；未关闭thinking、外层验收失败及有限覆盖见状态，不能称整体smoke已通过。

## 远端建议器

TypeSafe固定`https://api.typesafe.ai/v1`、`jev-1.14.0`；旧模板复用本地Jev的`/render`，迁移配置显式绑定独立CPU renderer。发送`state=rendered.context`、`questions.next_action={type:choice,instructions:rendered.question,criteria:…}`，criteria保留完整options。按[OpenAPI](https://api.typesafe.ai/openapi.json)校验model/answers/usage、选项全集和choice；概率和使用1e-5绝对容差，不归一化，并列保留服务choice。远端分词／内部prompt不同，本地32704预算不保证远端32k/question可接受；不加长上下文头，远端超长停止。

DeepSeek启用thinking/high、16384输出，不发temperature。正常tool_calls取合法首工具名；无调用且正常stop、有非空正文才为RESPOND。只注入工具名，不注入参数/思考；prefer允许probabilities=null，gate拒绝。`external_assistant_empty_reasoning`在每次请求前清除executor reasoning/think块，仅在assistant副本补`reasoning_content=""`；正文、调用/返回和原历史不变，不拼接建议器旧推理。`advisor_requests.history_compatibility`记录规则及处理数量。接口拒绝时停止，不自动重试。移动别名及响应model/fingerprint不能证明固定权重或等同ActionOPD训练教师。

上段动作规则的benchmark例外：API-Bank按官方文本抽取名称，Stable将正常无工具正文记为Thought并中性提示。`advisor_requests.output_protocol`分别记录`tool_calls`、`apibank_text`或`stabletoolbench`，保留完整原始响应和成本；DeepSeek的16384与执行器8192不构成等预算协议。

CPU测试用loopback假HTTP服务及SDK mock，须允许绑定本机临时端口，无外部推理。真实验收仍须覆盖TypeSafe近32704预算、长上下文和完整episode。ActionOPD按统一采样/8192/YaRN运行，不称论文greedy/4096协议复现；资产见PROJECT。

工程smoke模板固定tmp、1次、无配对推断。TypeSafe及DeepSeek原生工具调用路径的重名候选使用`candidate_0000`等位置身份，保存candidate_mapping并映回原名，执行器仍看到完整原工具表；概率建议继续按原名求和。API-Bank的DeepSeek文本协议不改名。DeepSeek仅在启用位置别名时，在请求副本末尾追加system说明：下一次调用须使用候选表里的精确别名，历史/描述中的原名仅供理解，仍可正常回复。历史调用不猜测映射、不改写，普通无重名请求不追加；返回原名仍按候选外名称处理，不放宽校验。ToolHop工具源码、API-Bank目标答案和ToolSandbox用户私有提示不进入建议器。

## PA-Tool：先冻结接口，再运行独立条件

原生Qwen3-4B thinking权重不变，只改工具与顶层输入参数名。实现依据[论文Algorithm1、Appendix A.3/L](https://aclanthology.org/2026.acl-long.948.pdf)，不是作者执行代码：pa_tool_prepare负责命名/选择，pa_tool_catalog复用官方转换，pa_tool映射结构化接口，pa_tool_text处理API-Bank确定文本跨度。生成用llm-efficiency/RapidFuzz，提取及评测用各benchmark原环境。

### 冻结的PA16方法

`configs/pa_tool_prepare.json`是批准的PA16项目变体：每工具/顶层参数16随机候选（T=0.4、top_p=1、top_k=-1、min_p=0）＋1个T=0参考，thinking、8192输出、20261015派生逐请求seed；17槽全尝试、不补抽。仅从最终content读单一ASCII标识符、最多64字符，不从reasoning取名。

原始provider可证实的length/invalid_final_name随机槽排除、不用原名补票；至少12随机合法且参考合法才选名，否则保留原名，记insufficient_valid_candidates/reference_failed，同时发生以后者为主、全部失败槽保留。Levenshtein邻域阈值=0.2×合法候选最大长度，保留重复频次；密度=其他合法邻居数/(合法数−1)，用精确分数跨有效数组件比较。排序为密度降序、参考距离升序、原槽首次出现；同轮争名最高密度锁定、同密度按原工具/属性序，输家移下一候选、不重算分数。回退原名先预留，其他候选避让；原名非法/重复或候选耗尽仍停止，不加后缀。

参数提示使用最终工具别名；工具名在可共现作用域唯一，参数只在本工具唯一。相同schema可复用候选，工具顺序/occurrence保留；BFCL目录含迟后释放工具，但每请求只展示当时可见候选。生成只见描述和参数的工具上下文，不见题目/答案/私有数据库/评分状态。

服务/网络/数据错误、缺必填、畸形响应、错模型、多choice仍显式失败，首例停止提交并收齐在途；成功槽须有合法名字，失败槽须有匹配provider证据，不能把名字缺失视为可排除失败。候选数、失败处理、归一化密度、同分规则是项目补充，旧32＋1资产/结果不重写、不自动转换，用对应冻结源码复现。

论文采用32候选，§3.1以目标工具使用模型命名，§5/Appendix C的Qwen实验为thinking mode，但未分开规定命名开关；Appendix L要求只输出名称不能推出关闭内部thinking。[公开main](https://github.com/holi-lab/PA-Tool)未提供可核实该开关的执行入口。命名thinking/8192及失败/seed/排序是披露的项目设置；8192自03a049a独立读取PROTOCOL.max_tokens、非执行器继承，是含thinking的单请求上限，不是名称长度/必用量。已有8192仅reasoning截断不证明预算必要/最优；降预算可能改变回退覆盖，须另定协议。只关命名thinking是诊断/变体；执行器T=0.6也不同于论文T=0，不称原样复现全部实验。

### 隐藏描述与接口边界

ToolSandbox批准`toolsandbox_hidden_description=preserve_current_visible_name`：catalog记录官方starting_context的TOOL_DESCRIPTION_SCRAMBLED/ARG_DESCRIPTION_SCRAMBLED；仅相应扰动下工具描述显式空串、或参数缺description时保留当前公开名、零命名请求，其余照常。不得读扰动前隐藏名称/说明；None、非法类型、无该扰动的缺描述及其他benchmark仍失败。保留名先预留并核冲突，mapping.description_preservations记唯一组件/原因，summary按组件/原因计数、上界扣这些请求，不造失败slot/usage；与generation_fallbacks分开，加载核验记录/实际名，缺新协议/来源标记的旧资产不升级。已批准边界smoke只加`wifi_off_3_distraction_tools_tool_description_scrambled`及`wifi_off_3_distraction_tools_arg_description_scrambled`各一次，不开放其他场景/改正式题集。

- 只改工具名、顶层输入键及其required引用；嵌套、输出、描述、政策、示例、普通业务值不改，未实现根属性引用显式失败。Stable先精确处理Finish控制接口，再按官方次序匹配业务后缀；Finish/参数/原调用字符串和官方未声明required原样保留。
- ToolHop优先顶层description；仅其缺失时array优先非空items.description。仍不适用的object/array按原序深度优先properties/items，换行连接非空子description原文，只用于顶层参数命名，不加原参数名/题目/隐藏实现、不摘要/补schema。显式空顶层、非法或全无描述失败；旧parameter_description_sources字符串保留，新来源记rule=descendant_descriptions与有序分段paths，是批准的输入补充。
- Stable参数显式空字符串description原样进入提示，只给工具别名/描述，不给原参数名、不补描述；summary.empty_parameter_descriptions单列，缺字段仍失败，控制接口除外。
- API-Bank ToolSearcher仅映射定义名/顶层输入键，普通返回不改。同名定义只有description/desc_for_search不同、其余format/输入/输出完全相同时共享别名并用唯一instruction初始定义命名，两份描述/来源保留；其他冲突或无唯一初始定义停止。文本双向转换用官方parser核完整名称/参数字典含值，重复顶层键沿官方后值覆盖，不改嵌套；不安全输出为接口违规，不安全历史投影为工程错。
- 未展示旧名/旧键或不能安全还原的调用记executor_response_error，终止为失败，原响应留成本账本；缺失/未完成mapping是工程错，不退native。执行/评分用canonical调用，provider请求/响应保留别名；评分恢复不调执行器。

API-Bank/ToolSandbox/Stable的PA接口失败为终态、official_scored=false，内部success=false只表示终止，不造官方0分或布尔连续分。按条件/类别报失败/计划/缺分，受影响类别完整均值null；execution_coverage_complete与coverage_complete分开，运行结束仍可评分partial。API-Bank把显式失败身份交worker，其他预测仍按发布序/L3共享manager评分；失败点不执行工具、不补空预测、不删题、不恢复重试。ToolHop计false例外见正式统计。

### 从零准备、冻结与成本

本轮正式资产统一用`prepare`从零生成，不导入历史账本/mapping，不接受--reuse-from/--resume；默认预览，临时catalog自动清理，真实生成保存catalog快照及原prepare命令，同一新运行内部相同组件缓存保留。prepare显式要求绝对可执行--catalog-python及--qwen-model-path，generate要求后者；原生qwen3-4b目录须与配置/catalog指同一目录，执行再核模型名/真实路径/上下文。预览不查服务、不要求模型文件已就绪，但catalog解释器须可用，不按Conda名称猜路径。新机BFCL/τ²/ToolHop/API-Bank可用经验证现代runner，ToolSandbox/Stable独立环境。

```bash
cd /data/wangsq/code/ActionDistill
JEV_PREPARE_PYTHON=/data/wangsq/software/miniconda3/envs/llm-efficiency/bin/python
JEV_CATALOG_PYTHON=/data/wangsq/software/miniconda3/envs/trust/bin/python
JEV_QWEN_MODEL_PATH=/data/wangsq/model/Qwen3-4B
JEV_PA_BENCHMARK=bfcl_mt
JEV_PA_CONFIG="experiments/jev_bench/configs/${JEV_PA_BENCHMARK}_baselines_formal.json"
CUDA_VISIBLE_DEVICES= LITELLM_LOCAL_MODEL_COST_MAP=True \
  "$JEV_PREPARE_PYTHON" -B experiments/jev_bench/scripts/prepare_pa_tool.py prepare \
  --config "$JEV_PA_CONFIG" --catalog-python "$JEV_CATALOG_PYTHON" \
  --qwen-model-path "$JEV_QWEN_MODEL_PATH"

# 仅在预算、资源和服务获准后由用户启动；不传旧命名来源。
JEV_PA_DIR="experiments/jev_bench/outputs/pa_${JEV_PA_BENCHMARK}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost TZ=Asia/Shanghai \
setsid --wait "$JEV_PREPARE_PYTHON" -B -u experiments/jev_bench/scripts/prepare_pa_tool.py prepare \
  --config "$JEV_PA_CONFIG" --output-dir "$JEV_PA_DIR" \
  --catalog-python "$JEV_CATALOG_PYTHON" --qwen-model-path "$JEV_QWEN_MODEL_PATH" \
  --concurrency "${JEV_PA_CONCURRENCY:?请指定获准并发}" --execute
```

可--concurrency16仅并行同组件slot，仍按槽选择；同模型第二端点用--additional-base-url并逐端点核身份/上下文，账本保留provider_base_url。具名候选失败继续17槽，服务/数据错误停止提交并收齐在途，不静默重抽；保留seed/采样不承诺并发逐token相同。

底层分离流程：catalog --execute --output <新tmp catalog>只提取公开接口；generate --catalog <catalog> --qwen-model-path <原生目录>默认预览，真实生成另给新--output-dir/--execute。**只有底层generate的同协议恢复/复用**可用：--resume --execute严格匹配已提交源码/catalog/协议/服务含附加端点；合格已停止记录也可--reuse-from <准备来源>导入新目录，与resume互斥。逐槽核组件、最终工具别名上下文、prompt、seed、实际采样、模型、响应/usage，已知成功不重抽，具名失败凭原始响应继承，未提交槽原位完成；一般失败/返回状态不明停止，不兼容旧32＋1、不重解释旧失败。此能力不是本轮正式prepare使用路径。

仅产catalog.json/mapping.json/summary.json、diagnostics/generation.jsonl及command/run_status。账本含started/finished、seed、原响应/usage/延迟；generation_fallbacks记组件/原名/原因/失败slot，naming_fallbacks按工具/参数及唯一组件计数；candidate_statistics记有效数/排除槽/参考合法，candidate_outcomes记工具/参数直方图及排除/参考失败。候选排除、原名回退、自然生成原名分别计数，回退不伪造输出；completed允许披露回退，不表示每请求成功。

preparation_cost只记资产全部历史＋新增准备成本一次，不并入每episode；new_preparation_cost为本次、reused_preparation_cost为继承，未知usage/未决新请求不补0。运行在metadata/run_status.pa_tool_assets按条件冻结mapping/catalog内容SHA256，恢复/每episode从同一已核字节解析，身份缺失或同路径换内容停止；纯评分恢复只用保存canonical输出，不重载mapping/调执行器。

`scripts/analyze_pa_candidates.py --output-dir <新目录>`为CPU历史重放：5份旧账本先用冻结源码逐字段复现32＋1映射，再比前16槽与seed20261015的1000组无放回子集；固定旧slot32参考/旧序/频次，同组件跨scope同抽样，不生成正式mapping。工具名变更而缺精确参数上下文记未知，碰撞依赖不可假定，耗尽/不可判/完整可判分列；旧提示token差不代表新总成本，子集分位数不是置信区间，真实slot16参考seed不等同旧slot32。

真实smoke须实际新mapping、只改接口、业务工具真执行、官方评分及完整成本/失败；similarity=1但零业务调用不能证明链路已验收。六份独立PA模板任务数见上表，旧66槽复用、全量公开目录CPU检查和PA16两样例结果见状态；CPU actor不证明候选合规或方法有效。

## 失败、恢复与结果

本节布尔结果/正式统计适用于BFCL/τ²，四项扩展不得套用。评分完成独立于`success`：API-Bank回复ROUGE-L、ToolSandbox similarity、Stable SoPR保持`success=null`并保存canonical连续值，是否已评分另记。

每个批次只有`metadata/command.txt`和`metadata/run_status.json`两份自动元数据。计划、配置、代码和服务身份被冻结；`command.txt`同时持有独占锁，禁止第二个协调器同时续跑。

`episodes.jsonl`以condition/category/task_id/trial/attempt标识结果，是汇总唯一输入；原始ID和领域保留。目录为`episodes/<condition>/<category>/<task_id的SHA256>/trial_N/attempt_N/`，存score、轨迹、episode.json及诊断，支持telecom长/特殊字符ID。汇总写metrics.json并共用批次锁；tmp与outputs的worker目录隔离。

- BFCL使用官方score.valid；τ²使用官方reward与is_successful函数，不另造成功代理。模型任务失败保留分母。
- 明确executor上下文超限和响应错误计具名失败；传输超时改为未解决service_error，不算模型答错。未知400、所有执行器SDK `InternalServerError`（含500/502/503/504）、编程及评分异常均显式停止；不伪装成上下文超限或可恢复service_error。BFCL参数/工具错误走官方流程；仅BFCL的score.json允许NaN/Infinity以保留官方状态诊断，网络、概率、成本及其他产物仍严格。
- Jev仅在HTTP 422且`error.type="ValueError"`、`error.message="over_length_after_drop"`精确匹配时，采用已批准的`over_length_after_drop_neutral`扩展：当前请求注入既有中性消息，继续官方循环，保留原错误和耗时。记录`status="neutral_fallback"`、具名`fallback`；`prediction/gate/first_action_matches`均为null，不伪造推荐或采纳，不计作置信度门控。每attempt及汇总的`cost.advisor.over_length_after_drop_neutral`单列次数；未知token/模型耗时/截断计入缺测，不补零。服务本身仍返回422，不改请求格式或重试。
- 建议器动作响应格式错误采用用户批准的`advisor_format_error_neutral`：保留原响应、错误、耗时和可验证成本，当前请求注入既有中性文本；prediction/gate/first_action_matches为null，单列次数。包括非法动作分布、非法首工具名、DeepSeek在原生tool_calls协议下用文本输出`[ApiName(...)]`及不完整动作响应；不把它们误算RESPOND。API-Bank明确采用官方文本调用协议，可解析的`[ApiName(...)]`是合法建议；非空正文无可提取调用按已确认规则记RESPOND。输入、身份、usage损坏、HTTP鉴权/连接/限流错误不属于此扩展。本地Jev的token数、截断计数/标记及模型耗时在动作解析前独立校验，缺失或非法均显式失败；仅动作损坏时仍保留已验证的元数据和成本。
- 其余Jev失败不静默取消建议。传输故障、429/5xx记service_error；其他确定性4xx及身份错误记advisor_error，保留HTTP状态和结构化原因，需诊断而不能原样resume。两者success=null，首次未解决错误即停止派发，不改成连续三次才停；SDK不自动重试。
- τ²模拟用户的SDK/API错误、非法响应（包括缺失usage子字段、不可用工具）、明确上下文错误或官方user_error均记为user_simulator_error，success=null；停止派发，修复后显式续跑。合法用户工具的执行错误仍走官方环境逻辑。SDK自动重试关闭，编程异常仍为fatal，不能借此自动重抽。
- 已评分结果或协议内模型失败不重跑；KeyboardInterrupt记interrupted。resume只恢复未完成推理；attempt.json持worker锁，旧worker存活即拒绝。已有完整trajectory/score只允许评分恢复，复用既有score，不重新抽样；未知推理异常不能借此恢复。
- resume核对jev_bench/src、scripts、实际配置；用advisor时另含serve_jev.py、jev_train/src/jev_train、dataset_tool_inventory/src及decider源码，保留原commit。文档/测试/无关配置不阻塞；官方checkout须匹配固定commit且相关源码/输入干净，仅忽略未跟踪pyc。正式执行及续跑的相关运行源码须已提交，配置/服务身份/温度不变；磁盘版本不证明常驻进程已更新。Jev异常可跨spawn传递；不可序列化SDK fatal在worker边界转普通异常，保留原类型/消息。

评分恢复：`<对应环境python> -B experiments/jev_bench/scripts/summarize.py --run-dir <运行目录> --rescore`。持批次/attempt锁、保存评分代码身份、原failure/attempt/成本；正式恢复要求源码已提交。完整轨迹缺失即拒绝，已评分结果不重算，缺失成本不猜测。全部覆盖才completed，否则无其他fatal时partial，可继续剩余推理。恢复不重跑执行器；API-Bank可能调用官方联网工具，Stable可能继续请求judge。单独summarize仅补索引/派生汇总，不调用模型。

failed批次须有纯评分故障证据：逐attempt用`error.scoring_attempts`；API-Bank在`phase=scoring`时可用非空`error.scoring_groups`，此时scoring_attempts仍须为列表但可空。相关原attempt/组全部恢复后才清除活动错误；其他fatal及缺ownership不猜测恢复。原错误归档`scoring_recovery.previous_errors`，不表示全部请求已解决。仅`phase=summarizing`且推理评分全覆盖时可走summarize-only，成功写metrics后completed并同步attempts，不借此转partial。

新提交的评分代码可恢复旧完整轨迹；剩余推理仍须原冻结源码，不改`state.code`或换run_id冒充续跑。ToolHop已接`score_saved`，复用已保存轨迹和官方评分，不重新请求actor。

四类按官方等权平均，精确双侧配对置换以任务为单位，保留任务内全部重复。成功率差按`treatment − reference`报告；95% percentile区间在每个类别内分别有放回重采样任务，保留同一任务的全部条件和重复，固定10,000次、seed=20261015。每次重采样都保持四类等权，不把重复运行视为独立任务。覆盖不完整时只报观测计数和成本，不给完整总分、区间或配对结论。

成本包含全部持久化attempt，分执行器、建议器、模拟用户。未知usage不补零，失败响应已知用量保留；本地Jev生成token确为0。TypeSafe的local_prompt_tokens与remote_input/output_tokens分开，DeepSeek reasoning_tokens属于completion_tokens，不重复加总；缺细分记缺测，不在代码内换算美元。

远端每次predict（含render前置失败）写`diagnostics/advisor_requests.jsonl`，提示/采纳另存advisor.jsonl；成本只读前者，不重复相加。advisor.requests计全部建议，provider_requests仅计provider_called=true；远端token/reasoning/模型耗时也只在已调用记录上统计，未调用不计缺测，调用后usage缺失仍记缺测。两种恢复共用账本，保留孤立请求和精确renderer超长计数。key及回显密钥脱敏；渲染耗时单列但已含在端到端latency中，不重复加总。

成本按显式advisor类型输出，不适用字段省略：none只有requests；jev_local无provider_requests/远端字段；TypeSafe无reasoning_tokens；DeepSeek无truncated/messages_dropped/local_prompt_tokens/render_latency_seconds及超长fallback计数。类型写入attempt.json供恢复，不从记录猜测；类型不符或同组字段集合不一致即报错。

账本落盘前中断以accounting_complete=false及attempts_with_incomplete_accounting披露，仅报告已观测成本。耗时/步数/工具数保留known/missing；native无advisor、BFCL无模拟用户为有效零。不同tokenizer不代表相同计算量，预检成功不证明推理可用。

已有运行可通过以下CPU入口重新生成该运行的派生汇总；正式产物的重新统计仍须先获授权：

```bash
python experiments/jev_bench/scripts/summarize.py --run-dir "<本项目tmp或outputs中的运行目录>"
```

## τ²协议与验收

τ²固定Verified本地commit，见PROJECT；其上游为`864350a8971a8f8ee9e7b8472e2edc380a806b0c`，另含未推送retail修复。校验导入路径/数据目录，TAU2_DATA_DIR冲突即失败。base为retail114/airline50/telecom114，不混旧retail40或telecom full2285。

显式使用官方普通双控、telecom manual policy、max_steps=200、max_errors=10、validate_communication=False、EvaluationType.ALL。max_steps计算官方消息路由步骤，max_errors按官方实现累计工具错误；评分默认不调用额外LLM judge。不增加旧retail代码的强化通信提示，也不删除正文与工具混合响应来规避协议。

Qwen沿用统一预算。τ²和ToolSandbox用户均为官方`https://api.deepseek.com`、`deepseek-flash`、非思考模式：`extra_body={"thinking":{"type":"disabled"}}`，temperature=0、max_tokens=8192、timeout=120秒，无reasoning_effort或自动重试。key读DEEPSEEK_API_KEY；这不是judge配置。ToolSandbox已开放本地场景tmp入口，真实验收范围见STATE。

Jev仅在agent决策点调用，接收agent的可见history及工具，不接收用户私有scenario、user tools、其工具返回、隐藏状态或GT。RESPOND仍表示向用户沟通，不能替代用户STOP或当作任务完成。用户工具调用留在完整轨迹中供审计。

τ²总分按任务等权，三域原任务占比为114/278、50/278、114/278，另报分域成绩及官方pass^k。success按官方is_successful(reward)判断；未评分的具名executor失败按批准口径计失败并单列。正式每题1次，仅支持pass^1，即成功率，不能称pass@k。成功率差仍做精确任务配对置换和10,000次、seed=20261015的域内任务配对bootstrap，保持原域任务比例；BFCL仍保持四类等权。模拟用户未解决时不输出完整总分与配对结论。

CPU回归覆盖三域官方成功/失败、telecom付款与状态同步及Jev不可见用户私有历史。真实smoke仍须检查身份、所有条件官方评分、建议/成本、reasoning传递和用户通信协议，并区分工具错误与非法模拟用户响应。除精确中性扩展外，服务失败不能退回native。

沿用官方提示、角色翻转和STOP。已有私有reasoning_content保留代码继续按原键穿过模拟用户自身的工具调用和后续轮次，非思考模式不要求返回该字段；不因换模式删除原始诊断，也不把用户私有历史送入执行器/Jev。temperature=0不保证不同远程请求逐字一致。

## ToolSandbox：本地场景执行与连续评分

官方角色会把工具调用 ID 拼入 Python 变量名。适配层仅在路由副本中，将非 ASCII 标识符或以保留前缀 `ts_` 开头的 ID 编码为 `ts_` 加原 ID 的 UTF-8 十六进制，避免 vLLM 的连字符 ID 导致语法错误；函数名、参数和原始响应不变。官方工具返回、历史重建及 reasoning 关联使用同一转换后的 ID，`response_normalizations.tool_call_id_to_python_identifier` 按参与者计数。

`tool_call_count`统计执行器生成的工具调用数，不包含用户的`end_conversation`。消息预算耗尽时，末批调用可能尚未交给环境执行，因此该字段不能解释为实际执行或成功完成的工具次数；当前字段及统计规则保持不变。

ToolSandbox独立环境及httpx0.27.2兼容约束按[迁移说明](../migration/README.md)重建，不向trust/mapika补装。CPU预览为2本地场景×native、user_simulator=null、mode=cpu_only；服务检查/执行在导入环境和创建批次前拒绝。main_smoke保留原本地子集，formal必须全量1032。

按官方named_scenarios和seed=42建完整catalog再选ID，保持工具顺序并恢复Python RNG。共129个原始场景×8变体=1032；由原始标签及7个固定后缀识别扰动家族，不按近似名称合并。官方类别重叠，不套BFCL互斥分组。构造场景不调用API。全量包含官方RapidAPI地图、天气、股票和汇率等工具，需要`RAPID_API_KEY`及对应订阅；地址、地图、当天/第4天天气和股票已有有界真实接口与官方解析验证。汇率按用户确认的服务迁移固定采用下述新接口；完整场景可用性尚未验收，有界接口检查不等于全量benchmark完成，状态与证据见PROJECT_STATE。

汇率环境变更：从`currency-converter18.p.rapidapi.com/api/v1/convert`迁移到`https://currency-conversion-and-exchange-rates.p.rapidapi.com/convert`，`X-RapidAPI-Host`同步改为`currency-conversion-and-exchange-rates.p.rapidapi.com`。原`from/to/amount`及大小写处理保留。新响应必须显式`success=true`、`result`为有限数值，零值有效；缺字段、错误类型、非有限值和服务失败均报错，不使用默认汇率、不重试或回到旧服务。项目层`toolsandbox_currency.py`在configure时安装精确汇率请求适配，将新数值映射到官方工具内部读取的`result.convertedAmount`，对外仍返回float；其他RapidAPI请求直接沿用官方helper。保留原converter对象、公开名称/签名/说明、ISO/Wi-Fi/凭据校验、schema扰动、PA别名及官方tool trace，第三方checkout仍固定原commit。所有本项目条件共用该后端；源码tree身份记录此次项目层环境变更，已有结果不重写，涉及汇率的历史新旧环境结果不得当作环境完全相同的运行混用。

`--check-services`和`--execute`在正式ToolSandbox模式下先检查`RAPID_API_KEY`非空；缺失、空串或纯空白均在任何服务请求及正式输出创建前失败。CPU预览和本地smoke不要求该key。检查只确认环境配置，不访问RapidAPI、不保存凭据，也不证明订阅有效；获准联网验收后再启动正式全量，不把缺凭据作为模型任务失败。

GuidedAgent/ConfiguredUser只替换官方model_inference，保留可见性、私有提示、候选扰动、执行和终止。合法空白正文按官方路由；仅字符串content+空tool_calls列表在路由副本转None，原响应保留，response_normalizations按角色计empty_tool_calls_to_none。非法正文/非空调用继续失败。每episode新建参与者/client，输出创建前拒绝复用；全局环境禁止线程间并行。建议不入后续历史，reasoning不进入模拟用户可见历史。

RESPOND表示agent回复user，只有用户end_conversation正常终止。官方30消息预算耗尽仍评分，标记message_limit。直接调用Scenario.play_and_evaluate，不用CLI所有异常计0包装；异常保留诊断/缺分并重抛，正式缺分处理见前文。tool_call_count统计执行器提出的调用，预算边界可能未执行，不称实际执行次数。

保留官方连续分数`similarity = milestone_similarity × 1[minefield_similarity == 0]`及完整mapping，0、部分分、1都是已评分结果，不转成`success`。完整状态来自`official/trajectories/<scenario>/execution_context.json`；`conversation.json`省略用户私有few-shot，不当作完整轨迹。执行器、advisor和模拟用户成本分开记录，失败请求也保留。

正式描述性类别汇总复用已有官方类别函数，遵循命名场景等权和scramble变体不重复计入three-distraction分类的规则；正式总体推断按家族整簇抽样/交换，见前文。CPU描述性fixture函数本身不生成CI或p值，不能把连续分数转成布尔成功。

CPU测试覆盖本地角色路由、并行工具、部分分/minefield、预算、可见性、门控及恢复。脚本actor分数仅验证工程；临时产物在本项目tmp，成功清理，未定位失败保留并打印路径。实际验证范围见PROJECT_STATE。

## API-Bank：发布逐点生成与官方组评分

API-Bank固定源码/版本、发布数据与MiniLM检索器路径见PROJECT。8个发布文件提供逐点任务；API准确率不是完整交互成功率。

| 发布任务 | API调用点 | API后的回复点 |
| --- | ---: | ---: |
| Level 1 | 399 | 375 |
| Level 2 | 135 | 103 |
| Level 3 | 245 | 50 |

共1307个发布决策点。L1/L2用官方Sample.from_chat_history和(file,id)，id不是行号/奇偶类型；L3用(sample_id,api_id)，回复api_id为末次API索引。内部ID用两字段紧凑JSON数组并保留原字段。245条L3 ICL仅是另一instruction，核对ID/input/output一致，不重复计任务。

发布instruction/input逐字保留，标签单独用于评分。Sample.apis扫描全对话是部分官方入口的设计，不能据此删除发布输入中已给出的工具，也不额外注入未显示的API。L1/L2从instruction和当前input的检索结果取可见证据；L2 API入口初始只给ToolSearcher。L3从input首行及已发生检索取候选，ICL示例工具不额外注入。

解析普通name/input_parameters/output_parameters及L3 apiCode/parameters/response两种schema，保留字段/来源。同名描述冲突记multiple_definitions及全部occurrence，不选描述、合并schema或将可选参数改必填。actor原样接system=instruction/user=input，不发native工具表；Jev/TypeSafe另接动作说明、完整公开材料及逐occurrence候选，保留参数描述，TypeSafe重名用位置ID，不从gold/未来补候选。DeepSeek使用下述原发布文本协议。近期调用仅解析明确User/API-Request，遇User重置，不伪造角色/调用ID或串联考题。

CPU评分明确采用与发布JSON点位一致的evaluator_by_json.py（L1/L2）和lv3_evaluator.py（L3），复用官方parser/manager/checker。另一个evaluator.py入口捕获范围更宽，不混合使用。保留L1/L2名称预检查及有限错误映射、L3关键词预处理和分阶段错误映射；未覆盖异常及损坏gold仍抛出。工具返回exception不自动判错，由checker判断。对预测/gold深拷贝隔离修改，CheckToken不补result.api_name，GetToday的显式input=null原样接受。

具名偏离`apibank_parameter_type_error`已获批准：L1/L2调用阶段，官方参数转换抛出`AssertionError("invalid parameter type.")`或`invalid parameter type. parameter: ...`时记为错误答案，reason为新增的`INVALID_PARAMETER_TYPE`，点级及`score_batch.protocol_events`计数。官方by_json在此处会抛异常；测试直接对照官方Evaluator，明确记录这一差异。不捕获checker中的同名断言，也不把未知schema类型、编程/环境错误统称参数错误；L3原错误映射不变。

回复沿官方规则：L1/L2空串为0、ROUGE-L保留4位；L3删除User:/AI:并strip，保留原始分。score_batch对单arm/trial按发布顺序评分且精确覆盖所选点：L1/L2每点新manager，L3共享一个，arm/trial间隔离。只有完整245个L3 API点/50对话才报告(50−错误对话数)/50的Sample Accuracy，子集不冒用此分母。正式入口要求六类全量1,307点。

DeepSeek建议器使用与actor相同的发布instruction/input，不传原生tools/tool_choice、不加建议器格式包装或位置别名。正常stop且无原生调用的非空正文，直接用固定官方get_api_call/parse_api_call抽取名称；实际提取出的名称必须属于公开候选（显式[RESPOND()]也不合法）。按用户确认，无可提取调用的正文记RESPOND，包括官方无法区分的漏括号文本；账本text_call_extracted记抽取情况，不猜测意图。参数既不执行也不额外校验；官方可抽取名称不代表参数合法。空输出、截断、原生tool_calls或候选外名称沿既定格式错误中性处理；其他服务/数据错误仍失败。Jev/TypeSafe仍使用原候选投影和公开材料包装，actor及官方评分不变，旧结果不重解释。

提示已指定调用API或回复，所以不能直接检验自然停止。分母仍为399/135/245；当前资产按(instruction,input)精确去重为389/118/245，不为凑论文数字改变规则。统计需考虑源对话相关性，去重分数最多另作敏感性分析。Translate/Dictionary的9个联网点不故意断网记零；仅在ALL_PROXY原先缺失时暂设空串解决上游恢复None的错误，退出恢复环境。

L1/L2错误标签沿官方error_statistic，MISS_INPUT_ARGUMENT对应缺required positional argument；L3保留具名本地阶段诊断。批准的参数类型偏离见上文，checker和其他未知异常仍抛出。

两个官方ToolSearcher的模型构造入口仅在CPU测试上下文中绑定到已有绝对路径、`device=cpu`及HF离线模式，保留官方检索算法；退出时恢复构造器、工作目录和环境变量。源码和模型文件不修改，绝不借远端模型ID自动下载。该上下文涉及进程全局状态，不在线程间并发使用。

cpu_preview为六类各1点且拒绝真实调用；main_smoke为六类×六条件，trust生成、jev-apibank worker评分。生成后pending_scoring，完整condition/trial才按发布顺序组评分，组score为唯一来源。恢复复用prediction/attempt；不完整组可继续缺失生成，不重抽已完成点。API与回复分开描述；评分可调用Translate等官方工具，不保证离线。

API-Bank服务预检和评分worker共用必填环境变量`JEV_APIBANK_SCORING_PYTHON`，没有默认解释器或自动环境切换。运行前指定目标机器上已按迁移说明重建的评分环境：

评分worker与服务预检共用必填`JEV_APIBANK_SCORING_PYTHON`，绝对路径须存在且可执行，没有默认或按环境名自动切换；缺失/无效在模型请求及worker启动前失败。`--rescore`需要新评分时也必须设置，普通CPU预览和仅复用既有canonical评分无需该变量。例如：

```bash
export JEV_APIBANK_SCORING_PYTHON=/data/wangsq/software/miniconda3/envs/jev-apibank/bin/python
```

独立评分环境保留Torch2.2.2、SentenceTransformers2.2.2与旧googletrans/httpx，不装新版OpenAI SDK；完整安装只维护于[迁移说明](../migration/README.md)。不要用trust替代或修改旧config/分数迁移。

## ToolHop：官方循环、评分与工程smoke入口

ToolHop固定HF快照见PROJECT，以下载元数据核对源码/数据，不另复制或要求Git checkout。995题保留原整数ID，内部用十进制字符串。actor只见question和原序tools.values()；含隐藏子问题的字典key、sub_task、functions、answer及答案类型标签均不公开。

保留3912个定义、60题重名及联合类型schema。300处required与Python必填参数不一致照原样保留，不改schema/源码或过滤任务；重名聚合见共享协议。Qwen3原生FC已接通，单题五条件真实联调及DeepSeek候选别名降级问题见[PROJECT_STATE](PROJECT_STATE.md)，不能据此推定完整题集通过。

每次独立加载官方evaluation_closed.py，仅替换模型钩子和在同命名空间编译获准超时捕获；执行sample_process，保留工具exec作用域及10秒装饰器。scenario/max_turns显式提供，当前选Free/9；Mandatory只是提示，不是tool_choice=required。达到上限仍执行末次工具，尾消息为tool得0；无调用（含null）结束回复。answer_correct同时检查最终答案和紧邻前一tool输出，不是严格最终答案EM。

具名偏离`tool_execution_timeout`：单次工具的`FunctionTimedOut`转为固定工具观察`{"error":"tool_execution_timeout"}`，保留tool_call_id和批内顺序，后续调用继续；每fixture的`tool_error_counts`与`tool_error_events`分别计数和记录工具名、ID、10秒上限。绝不序列化该异常的正文或参数（可能含隐藏函数源码）。普通工具异常仍走官方反馈；actor异常、KeyboardInterrupt/SystemExit不吞掉，不直接把整题补零。正式模式沿用已批准的具名失败及恢复规则。

CPU预览ID0/44×native×1。真实路径固定Qwen3 FC、thinking、8192输出、Free/9及官方零基turn；不是上游closed的0温度/256输出或open的Qwen2.5/1024复现。进入循环前检查工具import根模块，缺包显式失败；不能据此保证全部符号/工具可执行。

原四条件模板为两题×各1次，仅tmp，拒绝正式outputs/推断比较。传输/Jev/未知错误未解决并停止，只将官方answer_correct映为0/1，不套BFCL上下文失败计零。未知工具/非法JSON原样交官方反馈，循环只记录解析成功的dict参数。请求局部提示、SDK原文、概率/成本/截断/超时保留在诊断；内嵌评分前保存完整可见轨迹及scoring标记，不保存隐藏源码/target。并行批内中断时未返回观察不可恢复，轨迹partial、工具次数缺测，不用批前计数冒充全部。

ToolHop工程smoke仅报告覆盖、逐条件官方计分/正确数量、所有attempt成本与具名工具超时，不生成推断结论。formal模式按995任务配对计算准确率差、区间及精确检验，继承具名失败口径；最终模型条件仍待固定。

工具依赖固定Babel2.18.0、python-dateutil2.9.0.post0、dicttoxml1.7.16、holidays0.107、pytz2026.6、roman5.2、sympy1.14.0，传递依赖six1.17.0/mpmath1.3.0；安装与环境自检统一见迁移说明。

传递依赖为six 1.17.0、mpmath 1.3.0。task90顶层import触发pytz NameError，保留官方错误；Babel正向验证用内部import的task179，smoke仍选0/44。14题有顶层import（11题from datetime import datetime），不等于必败；290题引用七个依赖也不等于失败或修复覆盖数。

## StableToolBench：官方循环、缓存后端与DeepSeek评分

Stable固定源码及发布缓存版本见PROJECT；用jev-stabletoolbench轻环境，不导入训练pipeline。六组G1_instruction/G1_category/G1_tool/G2_instruction/G2_category/G3_instruction分别163/153/158/106/124/61题，共765；query_id全局唯一，与官方test_query_ids逐组对应，值0不是得分。

加载器保留query/api_list，relevant APIs仅用于评分。Archive.catalog/api_definitions直接复用固定官方fetch_api_json：原始category路径、首匹配、未匹配丢弃和原序均保留；已知43个候选位置/22题未匹配，不删除整题。Archive.openai_function复用官方转换，由调用方显式传入官方whitelist的standard_tool_name，保留末64字符alias、碰撞、参数重名及False/0/空默认值。tool_description允许显式None，不扩展到任意API description。

保留官方schema缺陷（如仅有required才输出optional、required含空字符串）。environment复用官方whitelist、None/空描述提示、reflect后写覆盖和endswith首匹配；candidate_diagnostics记原位置catalog_api_not_found。catalog_order必填且category连续成块，交错即拒绝。tmp及后续正式协议均固定ZIP类别首次出现及类内成员原序。single_chain/tool_calls converter使用CoT@1、12树深（非请求数）、观察前1024字符；actor空正文仍为Thought，普通文本不终止。Finish工具保留；Jev/TypeSafe未门控RESPOND只映Finish建议，参数由actor生成。

DeepSeek只有实际调用Finish才建议Finish；正常stop、非空正文且没有工具调用时按官方Thought处理，使用具名`stable_thought_neutral`中性提示。建议记录为raw_action=Thought、prediction/gate/first_action_matches=null，即使Qwen随后调用Finish也不算采纳；provider账本落盘前记录规则，cost及汇总独立计数，不计入格式错误。空输出、截断等仍走原错误边界，不把Thought当作任务完成。此修改仅适用于新运行，历史提示和评分保持原样。

真实链路保留官方wrapper的等待：每次后端请求前2秒，返回`Rate limit per minute error...`后再等待10秒。仅CPU测试显式替换等待函数；Finish等不请求后端的分支仍按官方实现处理。

同一只读Archive内按完整`catalog_order`缓存官方白名单，避免每题重复解压扫描全目录；目录顺序不同则分别构建，保留同名工具的后写覆盖、None描述及遗漏候选处理。不缓存执行环境或请求结果，不改变官方等待。

缓存约236MB压缩/2.61GB解压，只读ZIP。4070个候选位置引用的814个文档路径存在，不代表API可用。复用官方prepare_tool_name_and_url及str(parsed_input)精确键，保留顺序，不改排序JSON或查另一同名缓存。Archive.lookup为CPU只读接口；smoke顺序为cache→真实ToolBench API→官方check_result不通过后模拟。run内各条件共享cache/stabletoolbench追加层，flock/原子写入保护首次解析，示例顺序不变。无key在actor请求前阻塞，连接异常不静默模拟。坏工具JSON走官方virtual观察，经wrapper状态0，不改为2；cache_events和模型解析诊断独立保留。

白名单核对：全目录293个标准化名称重复，其中10组对应多个文件stem；任务引用59个重复名称、涉及126题，10组多stem中仅两组被任务引用。因此不能表述为“10个工具影响126题”。官方后写覆盖依赖目录顺序，已确认正式协议沿用上述smoke顺序。

评分复用官方ReinforceToolLearningEvaluator提示/解析、Unsure二次检查和最终Finish门槛，不调用另一is_passed规则；get_steps/缓存纯函数从固定源码提取，避免server及模块RNG重置。Finish不证明任务解决。CPU注入脚本judge；smoke用DeepSeek非思考模拟器(temperature=0/max_tokens=1024)和judge(0.2/1000)，独立计账；每题3次两阶段评分，Solved/Unsure/Unsolved为1/0.5/0。judge forced function、actor auto，不称官方GPT复现。缺字段/非法状态/截断输出失败不补零；SDK无重试，模拟器保留官方最多3次坏JSON重试。

CPU预览六组各1题×native，不调用真实judge；cpu_preview拒绝服务检查/执行，main_smoke仅tmp。独立轻环境与固定模块自检按迁移说明，不导入训练pipeline。

评分恢复持久化judge重复及成功阶段，从完整账本补checkpoint，仅请求未完成阶段，不重跑actor/tools；所有旧/新/失败成本从同一账本重建，tool_simulator与judge分列。服务端已执行、本地未落盘的崩溃仍可能重复调用。
