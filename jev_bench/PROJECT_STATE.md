# Jev benchmark 项目状态

更新时间：2026-11-04 16:17 CST +0800。

## 当前阶段

**当前准备迁移，正式主表未启动。** 六项工程入口、官方评分、恢复、成本及正式统计已接通并完成CPU验证；有界真实smoke不等于全矩阵通过或正式效果。最终Ours权重/温度/off或G1/G2及是否adaptive top3仍暂缓，PA全量映射未生成；不把smoke的A-38k/G1填成最终方法。

四项阶段性正式配置已准备：`configs/bfcl_mt_baselines_formal.json`、`configs/tau2_baselines_formal.json`、`configs/toolhop_baselines_formal.json`、`configs/apibank_baselines_formal.json`。条件依原序为Native、官方Jev、ActionOPD、DeepSeek建议器、STAR，comparisons空；每条件800/278/995题及1307决策点，五条件计划4000/1390/4975 episodes及6535点。PA仍属完整主表，待从零完整mapping；这些只是全量预览，未执行正式生成/评测，本轮不查端口或分配GPU。

迁移本地入口齐备：环境候选约束/安装、服务命令、目标机配置、tokenizer-only CPU renderer、Ours门控/top3、PA显式解释器和公开GitHub/私有ModelScope发布流程。唯一操作顺序见[迁移说明](../migration/README.md)。目标机尚未安装验证，新机OS/A100显存/驱动/CUDA12.8 SDK、CUDA扩展及目标机真实smoke仍待核对；候选约束不是已求解环境锁。完整训练池已在本机导出并验收，ModelScope尚未上传；本地主线清理、干净初始提交和CPU验收已完成；GitHub主线初始提交`43c2b44`已发布到远端`main`，旧开发历史未发布。本轮fork迁移改动已在本地验证，待同步至GitHub。

τ²修复已发布至 `https://github.com/DX0369/tau2-bench-verified` 的 `actiondistill-retail-item-fix` 分支，远端指向固定提交 `3d82db8f3579efe29609ab9daff058a9b7cb187c`。迁移配置保持同一提交，从fork直接恢复；ModelScope只交付训练数据包，不再承担源码交付。原有HEAD/runtime dirty检查保持不变。训练池、v2已交付38k但未训练的状态统一见[Jev状态](../jev_train/PROJECT_STATE.md)，不在本页维护第二份训练计划。

2026-11-04 14:30 CST +0800，两处差分oracle改为自包含请求合同，保留七类建议/门控场景、重复调用不累积建议、历史不变，以及Qwen/STAR的采样和可选工具字段验证。共享CPU回归168项通过；初次两项失败分别为本机Git不支持新init参数和沙箱loopback权限，修正测试命令、在允许本机端口的环境定向复验后通过。第三方固定版本检查通过；未调用外部模型、运行正式评测或改写旧结果。

2026-11-04 16:17 CST +0800，源码与数据交付分开：τ²直接从GitHub fork恢复原3d82db8提交，ModelScope源码上传/下载入口已退役；数据下载仅取冻结包12个文件，并保留私有可见性、固定revision及完整池校验。13项迁移CPU回归、生成Git命令的临时本地恢复及HEAD/runtime dirty检查、现有资产检查、CLI参数和文档语法检查通过。已联网核对fork分支SHA；未下载模型、调用模型API、运行训练或正式评测。

## 当前资产与CPU验证

固定资产/环境见[PROJECT](PROJECT.md)，精确方法和失败/评分合同见[README](README.md)。

| 范围 | 已确认内容及界限 |
| --- | --- |
| 服务与路径 | STAR已于2026-11-04 10:21 CST +0800同盘移至共享`/data/wangsq/model/STAR-4b`，48文件inode/大小/mtime不变，10配置仅改model_path。PA显式--catalog-python/--qwen-model-path，API-Bank显式JEV_APIBANK_SCORING_PYTHON，无Conda名/本机默认路径 |
| CPU renderer与策略 | 三条真实BFCL请求tokenizer-only回放与旧渲染完全一致，未加载权重；TypeSafe目标机无需旧A-38k权重。Ours默认top1，显式门控后可adaptive top3；Stable真实Finish/虚拟RESPOND各自概率/逻辑身份保留，不预选最终策略 |
| 迁移边界 | Stable自检实际导入七必需模块、缺失失败；本地τ²包检查及22项迁移/正式配置、27项配置/CLI回归通过。发布代码已处理SDK空下载成功、上传返回旧HEAD，远端发布仍待认证和revision核验；没有远端下载/上传或环境安装 |
| 六项工程与统计 | 全量公开输入预检、预览/官方循环/评分/恢复/成本及正式配置/整簇统计/Holm/缺分边界CPU验证通过；不证明GPU、全量外部服务或方法效果 |
| 共享失败语义 | 缺失/非法Jev元数据显式失败，只有动作损坏可advisor_format_error_neutral；精确renderer超长扩展、vLLM input_text/input_tokens识别与六项统一采样均有回归。Stable保持请求前2秒/限流后10秒；ToolSandbox计执行器提出调用，预算边界可能未实际执行 |
| 最近去重整理 | 19项建议器元数据/动作和24项ToolSandbox路由CPU测试通过；保留null/空tool_calls、空串/空白正文和官方调用计数。下载/部署归迁移页，文档收束未改输入/评分/失败合同、未请求模型/API或正式测评 |

首次PA测试误用缺rapidfuzz的trust、Stable/便携池loader缺测试路径、共享HTTP fixture受沙箱socket限制，已按既定环境/权限纠正后通过；这类CPU问题不当作真实模型失败或服务验收。8个无绑定IPC目录的24个socket、已退役BFCL工作树及4空目录按授权清理，共3,964,928 allocated bytes，4份原失败/完成记录和canonical目录身份不变；PA工作树/唯一账本继续保留。

## PA16：已完成有限真实验收，全量映射未生成

当前批准PA16：16随机＋1参考全尝试、thinking/8192、不补抽；至少12随机合法且参考合法才选名，失败排除/原名回退和归一化密度精确定义见README，旧32＋1资产不改写。正式prepare从零，不导入旧记录/不接受resume；底层同协议generate的严格恢复能力不等于本轮正式准备允许复用。

CPU命名重放 `outputs/analysis_pa16_replay_20261103_161730/`已逐字段复现5份旧32＋1mapping，来源账本/冻结源码保留；比较前16与1000组子集只分析、不产正式mapping。固定旧slot32参考/旧工具名，真实PA16参考slot16且seed不同；7个新参数上下文未知，旧提示抽样token差不代表新总成本、子集分位数不是CI，未碰撞/耗尽不证明全量无风险或方法等价。

全量公开catalog CPU检查覆盖ToolHop995题（142题172参数采用子描述，166 object/6 array）、API-Bank1307点（原61个level_2_response GetUserToken冲突修复），不调用模型或评分。PA16四项预览命名请求上界分别BFCL172550、τ²2414、ToolHop459697、API-Bank23256，尚未扣跨scope缓存。命名/重放、共享配置、六项边界等119项CPU回归通过；此前正式配置与命名83项回归亦通过，不能替代全量生成预算。

2026-11-03 17:11起GPU3两样例真实smoke已完成，源1d3a106、原生Qwen3-4B单服务、命名并发16、thinking/8192不变。API-Bank Calculator-level-1-3调用点2、ToolHop40各一次；18组件共306次全新命名（34+272），每组件16＋1完整，输出368687 token，零候选排除/参考失败/原名回退/服务错误/成本缺测，未复用历史记录。两mapping可由账本逐字段重建，请求prompt/seed/采样/模型/响应通过只读审计，两条唯一且全部官方评分。

- API-Bank实际`arithmetic_tool(formula='5*10')`还原Calculator并评分正确。
- ToolHop实际调用convention_center_finder(search_location/source)、digit_reverser，工具及location/data_source准确还原，下一轮历史投影一致；模型没查开业年份而反转2023答3202，官方判错，保留model_task_failure、不重试。

这是两项工程链路通过，不能称PA效果通过；候选失败排除/原名回退本轮未触发，仍仅CPU证据。有效19分31秒、GPU3/18013已退出、systemd success；首次空闲预检退出未发模型请求，原记录留存。证据为 `tmp/pa16_smoke_20261103_171147/`及对应`_pa_apibank/`、`_pa_toolhop/`。控制器原状态finished_pending_interface_audit，后续接口审计通过但不回写原状态/分数。

## 已完成工程接续与保留证据

| 真实验证范围 | 已确认事实、失败与修复边界 |
| --- | --- |
| 早期主表smoke | 2026-10-30端口冲突/IPC路径过长两次启动均无episode；后GPU4保存3条BFCL评分，因DeepSeek拒工具历史失败，τ²/ActionOPD未启动。GPU6按授权有界接续并独立恢复ActionOPD，原失败保留 |
| GPU7联调 | 有效40项全部官方评分、无重复、成本完整；修复前4条ToolSandbox工具ID路由失败轨迹保留。DeepSeek外部历史/候选位置别名有有限兼容证据，2次task44定点重放成功只覆盖已见故障。临时验收器评分路径误用及瞬时显存探针故障已定位，有效轨迹未重跑 |
| GPU2/3最小补测 | completed_with_failures：8计划、7唯一尝试、6评分、3成功。两份BFCL仅CPU恢复派生汇总，16个原证据文件未变。官方Jev显式代理补验τ² retail/0成功、8建议无错误/usage缺测；原HTTP451不覆盖，不能追溯证明已退出进程代理环境 |
| PA-BFCL与τ² | PA-BFCL独立恢复4/4评分、3成功；τ²三域均尝试，airline/retail评分，telecom调用用户侧check_apn_settings按既定PA接口合同失败、未评分。旧八小时队列、预算中断槽及未知usage保留，不重抽至成功 |
| STAR BFCL/τ² | 原7尝试6评分：BFCL4/4、1成功；τ²2/3评分、两reward=0，retail未关闭thinking而未评分。2条原请求token诊断复现，不从reasoning补工具/正文、不关闭thinking规避；模板/token归一化及DeepSeekR1解析器替换不消除此边界 |
| STAR扩展 | ToolHop2/2、API-Bank6/6、ToolSandbox2/2已评分，内部runner completed，但外层验收`'bool' object has no attribute 'values'`失败，不能称整体smoke通过；原证据 `tmp/remaining_smoke_20261102_013242*`保留。Stable未真实运行；最大观测输入35993 token/峰值37939MiB不保证完整131072窗口 |
| 旧PA ToolSandbox | 首次similarity=1但业务调用0，不算完整接口验收。用户确认增加官方remove_contact_with_id后，仅复用同schema的66成功槽、新建任务绑定，mapping字节不变、新增命名0；严格resume/绑定不放宽。实际删除/历史投影补验通过，similarity=0.7169，模型误读None而重复删除 |
| 旧PA其他链路 | API-Bank/ToolHop真实别名/参数还原已补覆盖，旧补验3条唯一评分、594命名。ToolHop620命名length、API-Bank漏括号/NO_API_CALL无接入缺陷证据；最新level_3_api [0,0]定位为模型不合官方括号格式，不推广全部条目、不补括号评分。取消槽/预算中断/未知usage不补零 |
| PA命名诊断 | 132请求thinking开/关合法数65/66、66/66，关模式两组32候选完全同名，仅是特定输入成本/多样性诊断，不改正式thinking。53项CPU回归与66旧thinking响应解析回放通过，后者无新请求/mapping/评分，不与五模型真实Jev前向回放混称 |

原native/四条件/main_smoke模板及独立原失败源继续保留；既有隔离工作区持有唯一τ²/BFCL账本、mapping和评分，不能整删tmp。历史共驻/CPU卸载/固定KV以原metadata为准，不代表常规部署逐token等价或当前资源状态。完整条件×领域×边界矩阵、Stable真实后端及正式方法效果尚未验证，不普遍重抽失败smoke。

## ToolSandbox联网接口与批准的汇率变更

保留最初未订阅HTTP403。订阅后地址、地图、当天/第4天天气、股票共5次HTTP200且官方解析通过，地图4地点、天气4天。原记录/复核为 `tmp/rapidapi_check_20261103_204849/evidence.json`、`audit.json`；天气一次性脚本误用归一化前字段仅CPU复核、原记录不改/无重发。当轮金融响应额度100/剩99，与介绍页200不同，不能按介绍页预算正式全量。

新旧汇率响应不兼容，经用户确认完成项目层精确URL/Host适配：新`currency-conversion-and-exchange-rates.p.rapidapi.com/convert`必须success=true且有限result、0有效，转换为官方内部结构；其他RapidAPI原样委托，无旧fallback/重试。官方名称/签名/schema/扰动/PA/trace及ISO/Wi-Fi/key检查不改，第三方checkout干净，所有本项目条件共用新后端，旧结果不重写。

36项ToolSandbox（含7汇率）和5共享边界回归通过；首次全回归因提前configure导致同名tests包冲突，修正测试入口后通过，未改环境。一次获准真实configure→官方缓存工具→新服务HTTP200，1 USD→0.89453 EUR，trace参数/结果核验，当轮额度1000/剩998；成功一次性脚本/响应清理，原失败保留、key不落盘。五类接口已有代表性实测，完整1032场景（129×8）、订阅持续可用性及正式预算尚未验收，无GPU或正式评测。

## 已知边界与待解决事项

- PA全量映射/完成率/真实准备总成本/正式效果未知，旧32＋1不转换。PA16失败排除/原名回退、ToolSandbox隐藏描述保留等未在本轮有限真实样例触发的边界仍以CPU证据标注。
- Stable仍缺TOOLBENCH_KEY，缓存未命中→真实后端→模拟器及judge链路未验收。43候选位置/22题缺API定义沿官方丢候选而不删题，目录顺序固定；六模板41参数中19空描述仅验证可处理，不证明无命名冲突。
- ToolHop七依赖/代表函数与pip check已验收；task90 pytz NameError属官方import作用域，原题/错误保留。14题顶层import、290题引用依赖不等于失败或修复覆盖数。
- STAR retail、PA telecom为已定位未评分，不算官方0；API-Bank漏括号/NO_API_CALL、命名length、预算中断和未知usage原样保留，不重试至成功、不关thinking/改官方解析规避。
- Jev元数据/身份错误不能中性降级；仅具名动作格式/精确超长扩展及Stable Thought中性沿README。缺分/成本缺测不补0，受影响类别完整均值null，有限DeepSeek兼容不证明全量合规。
- 目标机安装、CUDA扩展/A100 kernel、CPU renderer部署联调和新增top3真实smoke仍未验收；保持原采样/题集/评分/统计，不以CPU/import通过声称GPU可用。

## 简要历程

- 2026-10-27 22:45至2026-10-28 22:22 CST +0800：接通六项CPU、独立环境、远端建议器，冻结官方语义、成本与恢复，无正式结果。
- 2026-10-30 10:48至2026-10-31 16:32 CST +0800：保留两次无episode启动失败、DeepSeek历史拒绝、ToolSandbox路由故障；完成有界接续、Jev元数据/Stable等待修复及有限候选别名补验，不将旧失败改成通过。
- 2026-10-31 21:36至2026-11-02 18:54 CST +0800：STAR资产/真实有限覆盖、PA冻结与真实接口补验、GPU2/3最小补测及官方Jev代理补验完成；retail/telecom缺分、外层验收失败、预算中断和未知usage保留。
- 2026-11-02 20:05至2026-11-03 18:55 CST +0800：PA纳主表、扩展正式统计CPU验证、DeepSeek文本/Thought和缺描述合同完成；四项五基线/从零命名与PA16准备、旧账本CPU重放、两样例新命名/实际调用审计完成，ToolHop答错保留，无正式主表。
- 2026-11-03 20:26至2026-11-04 11:10 CST +0800：联网接口及批准汇率适配验收、迁移入口/CPU renderer/策略准备、STAR共享归位和授权临时项收尾完成。旧控制器/重复测试退役，HTTP fixture/评分解释器及测试路径问题按既定环境重跑通过；canonical资产、评分、账本、冻结源码和原失败不重写，未启动正式测评。
- 2026-11-04 12:16 CST +0800：文档收束为本项目PROJECT/STATE/README，部署/下载只维护迁移README；本轮未运行模型/API或科研统计。

## 下一步

1. 按唯一迁移说明完成认证、资产发布/恢复、干净环境安装和必要目标机验收，不普遍追加smoke或重抽旧失败；后续正式运行留目标服务器。
2. PA16两样例真实链路已验收，不重复普测。全量命名须目标路径/预算/授权明确后从零生成，少量mapping不能覆盖全量；不导入旧记录、不改写旧32＋1/原失败。
3. 最终Ours继续暂缓，不能默认A-38k/G1。纳Ours须显式最终权重/温度/门控及是否top3，纳PA须全量冻结mapping；配置核对提交后由用户启动。
4. DeepSeek API-Bank文本补验/只读核对已结束；Stable等ToolBench key。ToolSandbox五类接口及新汇率有限实测已结束，完整场景和正式预算仍未验收；保留官方解析、缺分和成本口径，不以局部通过替代正式结果。
