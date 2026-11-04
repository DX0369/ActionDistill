# Jev 标准 benchmark 评测

更新时间：2026-11-04 14:24 CST +0800。

## 研究目标

比较工具选择建议对官方任务表现与完整推理成本的影响。完整主表计划包括Native、TypeSafe官方Jev、Ours、ActionOPD、STAR、PA-Tool，DeepSeek建议器可选；主要比较为Ours对已纳入的其他条件。除ActionOPD和STAR外共用原生Qwen3-4B，比较各方法系统，不将系统差异归因单独训练。

六项适配和正式统计已接通，当前准备迁移，**正式主表未启动**。四项五基线配置已含DeepSeek、未含Ours；最终Ours权重/温度/策略暂缓，PA待从零生成全量映射。操作与科学合同见[README](README.md)，CPU/真实smoke覆盖、原失败及下一步见[PROJECT_STATE](PROJECT_STATE.md)，下载/环境/服务/目标机配置统一见[迁移说明](../migration/README.md)。

## 项目组织

`src/jev_bench/`维护配置、客户端、建议、成本与统计，`benchmarks/`保留各官方循环和评分；`scripts/`提供run/summarize、正式配置生成、PA准备和API-Bank隔离评分worker；`configs/`保存正式及有界模板，`tests/`与`tests_<benchmark>/`为共享/独立环境回归。`data/`保存发布输入和只读缓存，`tmp/`为验证/worker/诊断，`outputs/`为独立正式结果。

运行代码与CPU回归不依赖退役项目源码；请求注入和SDK字段使用自包含固定用例验证。Jev服务复用`../jev_train/scripts/serve_jev.py`与训练渲染；TypeSafe在目标机绑定独立tokenizer-only CPU renderer，不依赖旧训练权重。

## 资产与环境

本机工作区 `/data/wangsq/code/ActionDistill`，Conda前缀 `/data/wangsq/software/miniconda3/envs/`。模型服务为`llm-efficiency`（vLLM0.19.1）和`mapika`（本地Jev），通用下载模型位于共享`/data/wangsq/model/`；自训Jev保留在各run。以下为固定资产与当前环境，目标机候选约束尚待干净安装验证，不能当已求解环境锁。

| 资产 | 固定来源／环境 |
| --- | --- |
| BFCL | `third_party/gorilla/berkeley-function-call-leaderboard`，`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`；`trust` |
| τ² Verified | `third_party/tau2-bench-verified`，本地`3d82db8f3579efe29609ab9daff058a9b7cb187c`，含未推送retail属性修复；`llm-efficiency` |
| ToolSandbox | `third_party/ToolSandbox`，`c8571d7854316d2e1c5f288e59fe1e34e53f6dd1`；`jev-toolsandbox` |
| API-Bank | `third_party/DAMO-ConvAI/api-bank`，父仓`faf754671f5b850e1b4d1ff870db37a6a2480df6`；`data/API-Bank/test-data/`；`trust`生成、`jev-apibank`评分 |
| MiniLM检索器 | `/data/wangsq/model/paraphrase-MiniLM-L3-v2`；评分环境保留Torch2.2.2、SentenceTransformers2.2.2与旧googletrans/httpx |
| ToolHop | `third_party/ToolHop`，HF快照`b439d7279af359fda46e8117ae4f0245b75f5c6b`，按下载metadata核对；`jev-toolhop` |
| StableToolBench | `third_party/StableToolBench`，`aa4ed9f4737ad98bd706663f01d63623c3427812`；`jev-stabletoolbench` |
| Stable缓存 | `data/StableToolBench/server_cache.zip`，`stabletoolbench/Cache` revision`c6aca0c1cda74d54915eccabb60984dc083fb14c`；只读ZIP，不全量解压 |
| ActionOPD | `/data/wangsq/model/Qwen3-4B-ActionOPD-DeepSeek-V4-Flash`，`nics-efc/Qwen3-4B-ActionOPD-DeepSeek-V4-Flash` revision`edc207707325a40e16b7f09940352fd62c34b91a` |
| STAR | `/data/wangsq/model/STAR-4b`，`star-lab/STAR-4b` revision`cc429add770db952be0d40ec2c23cb63ccdf3ee7`；BF16、4,022,468,096参数，保留作者tokenizer/template及索引计数异常 |

TypeSafe固定`https://api.typesafe.ai/v1`、`jev-1.14.0`、`TYPESAFE_API_KEY`；DeepSeek固定官方`https://api.deepseek.com`、移动别名`deepseek-flash`、`DEEPSEEK_API_KEY`，不能声称固定权重或等同ActionOPD教师。Stable另需`TOOLBENCH_KEY`；ToolSandbox正式联网工具另需`RAPID_API_KEY`及相应订阅。密钥只从环境读取，不落配置/源码/日志。

PA准备显式传`--catalog-python`和`--qwen-model-path`，API-Bank评分显式设置`JEV_APIBANK_SCORING_PYTHON`，无本机默认路径。PA采用批准的PA16项目变体：16随机＋1参考、至少12随机合法且参考合法、thinking/8192；精确选择、保留隐藏描述、接口还原、成本和资产冻结合同只在README维护，旧32＋1资产不升级或改写。

## 运行与科研边界

- 入口默认预览；`--check-services`只检查已部署服务，`--execute`才运行，不负责启停。资源、资产、正式配置及授权明确后由用户启动，长任务使用既有管理器或setsid；localhost代理绕过只影响相关进程。
- Qwen/ActionOPD/STAR统一thinking采样、8192输出、131072 YaRN窗口；不是ActionOPD论文greedy/4096复现，不证明STAR完整长上下文效果，不混旧BFCL32k结果。
- Ours/TypeSafe共用公开候选摘要和历史预处理，不声称服务内部输入相同；DeepSeek一步建议与PA接口改名各有独立方法边界。建议器不接收隐藏状态、未来标签或执行器私有推理。
- Ours须显式选off/G1/G2，默认top1；adaptive top3需显式启用0.5阈值、先门控后展示、不重归一化。Stable真实Finish与虚拟RESPOND保留独立逻辑身份/概率，均映射终止动作；当前只是CPU工程入口，不代表最终策略已选。
- 各官方指标、配对单位、统计与失败/缺分独立保留；ToolSandbox similarity、API-Bank回复ROUGE-L和Stable SoPR不转布尔成功。未知成本与缺分不补0，不重抽已评分模型失败。
- ToolSandbox所有条件共用已批准的新汇率后端，旧结果不改写；密钥非空不证明订阅/服务可用。Stable评分恢复可请求judge，API-Bank可执行官方联网工具，不能称所有恢复均离线。
- 固定正式全量、每题每条件一次、配置/代码/服务身份/温度与PA资产。smoke/正式分目录，原失败与唯一证据保留；预览、CPU回归及有界真实联调不能替代全量效果结论。
