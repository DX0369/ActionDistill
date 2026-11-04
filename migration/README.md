# ActionDistill 迁移与运行

更新时间：2026-11-04 16:12 CST +0800。

本页是唯一迁移说明，覆盖旧机交付和新机运行。工作区布局为 `ActionDistill/experiments/` 代码仓库，旁边按 [assets.json](assets.json) 恢复固定版本的 `third_party/`，第三方源码独立管理，不复制进ActionDistill主仓库；下载模型、数据和实验输出独立管理。代码发布到公开 `DX0369/ActionDistill`，整理数据发布到私有 `dx0369/ActionDistill`。

| 使用路线 | 入口 |
|---|---|
| 旧机交付 | [代码与测评资源](#代码与正式测评资源)、[冻结训练池](#冻结训练检查点)、[发布](#发布与目标机下载) |
| 新机只测五基线 | [公共准备](#新机公共准备) → [基线测评](#一基线测评) |
| 训练Jev | 公共准备 → [选样、双卡训练与校准](#二训练-jev) |
| Ours已准备好 | 公共准备 → [服务](#服务部署配置和建议策略) → [联合测评](#三ours-联合测评) |

Ours就绪时直接选择联合测评；五基线批次用于先测基线。命令使用Bash，正式导出、安装、下载和实验由用户启动。目标机环境/GPU、完整池和最终Ours/PA资产的当前验收状态见[训练状态](../jev_train/PROJECT_STATE.md)和[测评状态](../jev_bench/PROJECT_STATE.md)。

| 入口 | 行为 |
|---|---|
| `prepare.py` | 打印源码、数据、模型恢复命令；`--check`只读检查源码/官方数据 |
| `environments.py` | 打印环境计划；`--execute`才安装 |
| `services.py` | 只打印服务启动命令 |
| `deploy.py` | 写新目标机配置，不执行评测 |
| `publish.py` | 打印发布/下载命令；`--execute`才远端写入或下载 |

## 代码与正式测评资源

`assets.json` 记录六个源码仓库、三套非 Git 数据、五个主线下载模型，以及六项正式范围。训练池、离线保留集和官方 benchmark 是三类不同输入；benchmark 保持官方格式和原读取位置。

| Benchmark | 范围 | 主要资源 |
|---|---:|---|
| BFCL | 800题 | 固定 gorilla 的四类 multi-turn 数据、答案、工具及评分 |
| τ² | 278题 | 修复后的 checkout、三域任务、DB、policy 和用户提示 |
| ToolHop | 995题 | HF 数据、closed 执行代码、两份版本 metadata |
| API-Bank | 1,307个决策点 | HF 发布输入、DAMO 工具库和数据库、MiniLM |
| ToolSandbox | 1,032场景 | 固定源码生成的场景、工具、评分及项目汇率适配 |
| StableToolBench | 765题 | 六组问题/ID、官方循环与评分、原始缓存 ZIP |

这些计量单位和官方指标不同。源代码恢复不代表外部服务、模型或正式运行已经验收。

在旧服务器只读检查已下载源码/数据：

```bash
cd /data/wangsq/code/ActionDistill
python -B experiments/migration/prepare.py --workspace-root "$PWD" --check
```

清单中的版本检查沿用正式执行器的 HEAD、runtime dirty 和 ToolHop metadata 规则。τ² 从 [DX0369/tau2-bench-verified](https://github.com/DX0369/tau2-bench-verified) 恢复，固定提交 `3d82db8f3579efe29609ab9daff058a9b7cb187c` 已包含 retail 商品查找修复。所有源码统一从 GitHub 拉取固定提交；原始教师轨迹留在旧机。

`prepare.py` 的 `sources`、`benchmarks`、`models` 可分别打印；Git目标目录须不存在，HF CLI保留版本metadata并支持续传。共享下载模型使用显式模型根目录，本机为 `/data/wangsq/model/`；自训Jev跟随run保存。

## 冻结训练检查点

当前池的科学边界是：旧 capped 输入，C1 后244,595候选，轨迹去重后150,234状态；默认 v2 支持150,129状态。完整池保留105条零参考单元，默认选样仍排除它们。去重按 `(来源,任务组,教师)` 保留一条轨迹的全部合格状态，不是每轨迹只取一个状态。

检查点保留训练正文原样，将轨迹、领域、回复来源/边界、R/O资格和审批事实放在独立注释中。`selection_reference.json` 保留原接受记录及原路径作为溯源；新机器不打开这些旧路径。默认参考仍是冻结 R00 状态计数。

```text
manifest.json
README.md
pool/train.jsonl
pool/annotations.jsonl
pool/selection_reference.json
eval/{val,test_id,test_ood}.jsonl
selections/task_reply_v2_38k_ids.txt
recipes/task_reply_v2_38k.json
metadata/{command.txt,run_status.json}
```

完整冻结池已导出至 `jev_train/outputs/selection_pool_20261104_124748/`，150,234条，正文读回、现有38k逐条一致、8k/38k/76k黄金重放和文件校验均通过，可直接进入发布。原manifest保留构造提交身份，公开初始提交不替代它。

以下命令仅用于以后获准重新构造冻结包；本次交付无需重跑。它是正式资产生成，由用户启动，输出必须是新目录：

```bash
conda activate llm-efficiency
cd /data/wangsq/code/ActionDistill
JEV_POOL_OUT="experiments/jev_train/outputs/selection_pool_$(date +%Y%m%d_%H%M%S)"
setsid --wait python -B -u experiments/jev_train/scripts/export_selection_pool.py \
  --preview experiments/jev_train/outputs/curation_preview_task_reply_v2_76k_20261103_205916 \
  --samples-38k experiments/jev_train/outputs/samples_task_reply_v2_38k_20261103_220325 \
  --eval-samples experiments/jev_train/outputs/samples_eval_normalized_20261025_225810 \
  --tokenizer /data/wangsq/model/Mapika-decider-2b \
  --output "$JEV_POOL_OUT"
```

正式导出会验证全部正文读回、已交付38k逐ID完整字段，以及8k/38k/76k的有序ID、配额和既有完整统计。评测输入绑定原规范化版本；不重新渲染、调用模型标注或重新匹配原始回复。失败保留目录和状态，不覆盖或自动重抽。

有界工程验证在同一命令追加 `--smoke`，并改用 `jev_train/tmp/` 下独立目录。它只从已有索引读取代表正文，读取既有计划和摘要，不调用全量配方重放；复制每个评测split前两行。包标为 `smoke=true`，不能用于正式训练，不能证明全量黄金回归已通过。

## 发布与目标机下载

公开主线在本地 `mainline` 分支维护，旧开发历史仅保留在本机 `archive/pre-mainline`、`jev-dataset-review` 等本地分支。以下入口要求显式指定源码分支，检查干净工作区、该分支可达历史中的常见凭据模式及指定origin，然后只推送 `mainline → main`，不推送旧分支或标签。认证使用已有Git credential helper：

```bash
cd /data/wangsq/code/ActionDistill
python -B experiments/migration/publish.py --stage code --source-branch mainline --branch main
# 核对后执行同一命令并追加 --execute。
```

交互终端也可在核对提交与认证后执行下列推送；凭据由Git认证提供：

```bash
cd /data/wangsq/code/ActionDistill/experiments
GIT_TERMINAL_PROMPT=1 git push --set-upstream origin mainline:refs/heads/main
```

数据上传使用 `modelscope==1.34.0` 的解释器，通过环境变量 `MODELSCOPE_API_TOKEN` 认证，命令和 URL 不含 token。入口要求远端仓库已经存在且为私有，不自动创建或修改可见性。本次按已确定范围私有迁移，不再增加来源审批步骤；数据包保留原有来源信息和上游使用条件。

```bash
cd /data/wangsq/code/ActionDistill
(
  set -e
  JEV_PUBLISH_PYTHON=/data/wangsq/software/miniconda3/bin/python # modelscope==1.34.0
  JEV_POOL_OUT=experiments/jev_train/outputs/selection_pool_20261104_124748
  read -r -s -p 'ModelScope token: ' MODELSCOPE_API_TOKEN
  printf '\n'
  export MODELSCOPE_API_TOKEN
  setsid --wait "$JEV_PUBLISH_PYTHON" -B -u experiments/migration/publish.py \
    --stage data --dataset-id dx0369/ActionDistill --pool "$JEV_POOL_OUT" --execute
)
```

池发布仅接受 completed、非 smoke 的150,234条包，并要求38k逐条一致、三档黄金重放及文件内容验证通过。上传前自动执行文件内容校验，只上传manifest列出的数据文件和两份metadata，不上传原始149GB轨迹、缓存或旧输出。记录最终40位revision；新机下载仅包含这份数据包，不依赖源码附件。

## 新机公共准备

### 路径、源码与下载

公共准备采用覆盖训练与六项测评的完整环境。先填写已有 Python ≥3.10、Conda 和 CUDA 12.8 SDK 的绝对路径；源码恢复不需要ModelScope token或数据包revision。代码目录固定放在工作区的 `experiments/`；模型根目录可自行指定。

```bash
export JEV_WORKSPACE_ROOT="$HOME/code/ActionDistill"
export JEV_MODEL_ROOT="$HOME/model"
export JEV_ENVS_ROOT="$HOME/.conda/envs"
export JEV_POOL_ROOT="$JEV_WORKSPACE_ROOT/data/selection_pool_v2"
export JEV_TRAIN_PREFIX="$JEV_ENVS_ROOT/actiondistill-train"
: "${JEV_BOOTSTRAP_PYTHON:?填写已有Python的绝对路径}"
: "${JEV_CONDA_EXE:?填写conda可执行文件的绝对路径}"
: "${JEV_CUDA_HOME:?填写CUDA12.8 SDK的绝对路径}"

mkdir -p "$JEV_WORKSPACE_ROOT"
git clone https://github.com/DX0369/ActionDistill.git "$JEV_WORKSPACE_ROOT/experiments"
cd "$JEV_WORKSPACE_ROOT"
export JEV_BOOTSTRAP="$JEV_WORKSPACE_ROOT/experiments/jev_bench/tmp/download_bootstrap"
"$JEV_BOOTSTRAP_PYTHON" -m venv "$JEV_BOOTSTRAP"
"$JEV_BOOTSTRAP/bin/python" -m pip install 'huggingface_hub==1.33.0' 'modelscope==1.34.0'
export PATH="$JEV_BOOTSTRAP/bin:$PATH"

"$JEV_BOOTSTRAP/bin/python" -B experiments/migration/prepare.py \
  --workspace-root "$JEV_WORKSPACE_ROOT" --model-root "$JEV_MODEL_ROOT" --stage all
```

## 环境重建

`envs/requirements/` 只列运行依赖，`envs/constraints/` 固定关键版本；它们是候选安装约束，并非已求解的 lock。目标为五个长期环境，其中 `train-runner` 合并训练/本地 Jev 服务/CPU renderer/现代 runner。若实际 resolver 或 smoke 证明无法合并，分别安装 train、runner，成为六个；不静默改包版本。

| 角色 | Python | 职责 |
|---|---|---|
| train-runner | 3.11.16 | 训练、离线评测/校准、Jev/renderer；BFCL、τ²、ToolHop、API-Bank生成和PA命名 |
| vllm | 3.11.15 | Qwen、ActionOPD、STAR执行器 |
| apibank | 3.10.22 | MiniLM检索、旧googletrans/httpx及官方评分 |
| toolsandbox | 3.11.18 | 固定旧SDK栈、官方场景与评分 |
| stabletoolbench | 3.10.22 | Pydantic 1轻运行栈、官方循环/评分 |

训练/vLLM固定Torch2.10 cu128；训练的causal-conv1d扩展需要匹配CUDA12.8 SDK或明确指定兼容wheel。目标机先核对系统、A100显存、驱动、`nvcc --version`，不复制旧环境二进制，也不默认把双卡训练改成八卡。

逐条执行上面打印的源码恢复和下载命令后，再安装以下环境。`environments.py` 不带 `--execute` 时仅打印计划，带该参数才安装，prefix须为新目录。安装会执行 `pip check`、导入来源和causal/FLA入口检查，并记录环境内 `resolved-packages.txt`；真实GPU前反向、保存恢复和服务smoke在目标机验收。

source模式禁用pip缓存；已有兼容wheel时，可将 `--causal-mode source --cuda-home ... --build-dir ...` 换为 `--causal-mode wheel --causal-wheel /绝对路径/causal_conv1d-1.7.0-兼容文件.whl`，两组参数互斥。Stable自检包括numpy、OpenAI、HTTPX、Pydantic、Requests、PyYAML和Tenacity。

```bash
"$JEV_BOOTSTRAP/bin/python" -B experiments/migration/prepare.py \
  --workspace-root "$JEV_WORKSPACE_ROOT" --check
"$JEV_BOOTSTRAP/bin/python" -B experiments/migration/environments.py --role train-runner \
  --conda-executable "$JEV_CONDA_EXE" --prefix "$JEV_TRAIN_PREFIX" \
  --workspace-root "$JEV_WORKSPACE_ROOT" --causal-mode source --cuda-home "$JEV_CUDA_HOME" \
  --build-dir "$JEV_WORKSPACE_ROOT/experiments/jev_train/tmp/causal_a100_build" --execute
for JEV_ROLE in vllm apibank toolsandbox stabletoolbench; do
  "$JEV_BOOTSTRAP/bin/python" -B experiments/migration/environments.py --role "$JEV_ROLE" \
    --conda-executable "$JEV_CONDA_EXE" --prefix "$JEV_ENVS_ROOT/actiondistill-$JEV_ROLE" \
    --workspace-root "$JEV_WORKSPACE_ROOT" --execute
done
export JEV_TRAIN_PYTHON="$JEV_TRAIN_PREFIX/bin/python"
export JEV_RUNNER_PYTHON="$JEV_TRAIN_PYTHON"
export JEV_VLLM_PYTHON="$JEV_ENVS_ROOT/actiondistill-vllm/bin/python"
export JEV_APIBANK_SCORING_PYTHON="$JEV_ENVS_ROOT/actiondistill-apibank/bin/python"
"$JEV_APIBANK_SCORING_PYTHON" -m nltk.downloader punkt
```

若环境按 train/runner 拆开，把 `JEV_TRAIN_PYTHON` 和 `JEV_RUNNER_PYTHON` 分别设为相应解释器。以后新开终端时重新设置上述路径变量即可。

## 服务、部署配置和建议策略

五基线和联合测评使用以下服务，仅训练时跳过本小节。先指定各执行器的 GPU 及显存比例；端口是本页示例，可成组修改。每次 `services.py` **只打印**一条启动命令，复制到各自终端或已有进程管理器中启动。

```bash
cd "$JEV_WORKSPACE_ROOT"
JEV_QWEN_PORT=8001
JEV_ACTIONOPD_PORT=8005
JEV_STAR_PORT=8006
JEV_RENDERER_PORT=18768
export JEV_QWEN_URL="http://127.0.0.1:$JEV_QWEN_PORT/v1"
export JEV_ACTIONOPD_URL="http://127.0.0.1:$JEV_ACTIONOPD_PORT/v1"
export JEV_STAR_URL="http://127.0.0.1:$JEV_STAR_PORT/v1"
export JEV_RENDERER_URL="http://127.0.0.1:$JEV_RENDERER_PORT"
JEV_SERVICE_ARGS=(--workspace-root "$JEV_WORKSPACE_ROOT" --model-root "$JEV_MODEL_ROOT")

"$JEV_RUNNER_PYTHON" -B experiments/migration/services.py --kind renderer \
  --python "$JEV_TRAIN_PYTHON" "${JEV_SERVICE_ARGS[@]}" --port "$JEV_RENDERER_PORT"
"$JEV_RUNNER_PYTHON" -B experiments/migration/services.py --kind qwen \
  --python "$JEV_VLLM_PYTHON" "${JEV_SERVICE_ARGS[@]}" --port "$JEV_QWEN_PORT" \
  --gpu "${JEV_QWEN_GPU:?指定Qwen GPU}" --memory-utilization "${JEV_VLLM_MEMORY:?指定显存比例}"
"$JEV_RUNNER_PYTHON" -B experiments/migration/services.py --kind actionopd \
  --python "$JEV_VLLM_PYTHON" "${JEV_SERVICE_ARGS[@]}" --port "$JEV_ACTIONOPD_PORT" \
  --gpu "${JEV_ACTIONOPD_GPU:?指定ActionOPD GPU}" --memory-utilization "$JEV_VLLM_MEMORY"
"$JEV_RUNNER_PYTHON" -B experiments/migration/services.py --kind star \
  --python "$JEV_VLLM_PYTHON" "${JEV_SERVICE_ARGS[@]}" --port "$JEV_STAR_PORT" \
  --gpu "${JEV_STAR_GPU:?指定STAR GPU}" --memory-utilization "$JEV_VLLM_MEMORY"
```

CPU renderer只加载固定Mapika tokenizer，复用32704预算、模板、截断和候选位置身份；`/render`提供输入整理，`/advise`拒绝预测。官方Jev预测由TypeSafe API完成，Ours使用下文独立本地Jev服务。vLLM入口固定BF16、TP1、max-num-seqs=1、131072 YaRN及既有解析器。

每次选择一个 benchmark，并在本节之后的同一终端生成配置和运行。正式题集由原生成器固定，切换 benchmark 不改变其评分协议。

```bash
JEV_BENCHMARK=bfcl_mt # tau2 / toolhop / apibank / toolsandbox / stabletoolbench
case "$JEV_BENCHMARK" in
  bfcl_mt|tau2|toolhop|apibank) JEV_BENCH_PYTHON="$JEV_RUNNER_PYTHON" ;;
  toolsandbox) JEV_BENCH_PYTHON="$JEV_ENVS_ROOT/actiondistill-toolsandbox/bin/python" ;;
  stabletoolbench) JEV_BENCH_PYTHON="$JEV_ENVS_ROOT/actiondistill-stabletoolbench/bin/python" ;;
esac
mkdir -p experiments/jev_bench/tmp
```

在运行终端提供 `TYPESAFE_API_KEY`、`DEEPSEEK_API_KEY`；StableToolBench 另需 `TOOLBENCH_KEY`，ToolSandbox 正式全量另需 `RAPID_API_KEY` 及已订阅服务额度。API-Bank 使用上面指定的评分解释器。`--check-services` 检查本地身份，也会检查 DeepSeek API 连通性；远端推理和官方评分由真实 smoke 验收。

## 一、基线测评

下面保留五基线，先用原模板的有界题目验收，再生成全量配置。更换 benchmark 时重做本节，使用新的标识。

```bash
JEV_BASELINE_TAG="${JEV_BENCHMARK}_baselines_$(date +%Y%m%d_%H%M%S)"
JEV_BASELINE_ARGS=(
  --template "experiments/jev_bench/configs/${JEV_BENCHMARK}_main_smoke.json"
  --workspace-root "$JEV_WORKSPACE_ROOT" --model-root "$JEV_MODEL_ROOT"
  --qwen-url "$JEV_QWEN_URL" --actionopd-url "$JEV_ACTIONOPD_URL"
  --star-url "$JEV_STAR_URL" --renderer-url "$JEV_RENDERER_URL"
  --conditions native official_jev actionopd deepseek_advisor star
)
JEV_SMOKE_CONFIG="$JEV_WORKSPACE_ROOT/experiments/jev_bench/tmp/${JEV_BASELINE_TAG}_smoke.json"
"$JEV_RUNNER_PYTHON" -B experiments/migration/deploy.py "${JEV_BASELINE_ARGS[@]}" \
  --output "$JEV_SMOKE_CONFIG"
env LITELLM_LOCAL_MODEL_COST_MAP=True "$JEV_BENCH_PYTHON" -B \
  experiments/jev_bench/scripts/run.py --config "$JEV_SMOKE_CONFIG"
env LITELLM_LOCAL_MODEL_COST_MAP=True NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  "$JEV_BENCH_PYTHON" -B experiments/jev_bench/scripts/run.py \
  --config "$JEV_SMOKE_CONFIG" --check-services
setsid --wait env LITELLM_LOCAL_MODEL_COST_MAP=True \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  "$JEV_BENCH_PYTHON" -B -u experiments/jev_bench/scripts/run.py \
  --config "$JEV_SMOKE_CONFIG" --run-id "${JEV_BASELINE_TAG}_smoke" --execute
```

核对实际工具执行、官方评分、建议器身份和成本记录后，生成并提交本批正式配置，再执行：

```bash
JEV_FORMAL_CONFIG="$JEV_WORKSPACE_ROOT/experiments/jev_bench/configs/${JEV_BASELINE_TAG}.json"
"$JEV_RUNNER_PYTHON" -B experiments/migration/deploy.py "${JEV_BASELINE_ARGS[@]}" \
  --formal --output "$JEV_FORMAL_CONFIG"
"$JEV_BENCH_PYTHON" -B experiments/jev_bench/scripts/run.py --config "$JEV_FORMAL_CONFIG"
git -C "$JEV_WORKSPACE_ROOT/experiments" add "$JEV_FORMAL_CONFIG"
git -C "$JEV_WORKSPACE_ROOT/experiments" commit --only -m "Configure $JEV_BASELINE_TAG" -- "$JEV_FORMAL_CONFIG"
setsid --wait env LITELLM_LOCAL_MODEL_COST_MAP=True \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  "$JEV_BENCH_PYTHON" -B -u experiments/jev_bench/scripts/run.py \
  --config "$JEV_FORMAL_CONFIG" --run-id "$JEV_BASELINE_TAG" --execute
```

结果位于 `experiments/jev_bench/outputs/$JEV_BASELINE_TAG/`，查看 `metrics.json` 和 `metadata/run_status.json`。这里只产生基线批次；Ours 的同批正式比较见第三条路线。

## 二、训练 Jev

### 便携配比与训练

完整池revision只包含验收通过的数据包，τ²修复源码从GitHub固定提交恢复。仅做benchmark不需要执行本节。下载工具沿用公共准备中的 bootstrap 环境。

```bash
cd "$JEV_WORKSPACE_ROOT"
read -r -s -p 'ModelScope token: ' MODELSCOPE_API_TOKEN
printf '\n'
export MODELSCOPE_API_TOKEN
"$JEV_BOOTSTRAP/bin/python" -B experiments/migration/publish.py --stage data-download \
  --revision "${JEV_DATA_REVISION:?填写完整数据包的40位revision}" \
  --destination "$JEV_POOL_ROOT" --execute
unset MODELSCOPE_API_TOKEN
JEV_SELECTION_OUT="$JEV_WORKSPACE_ROOT/experiments/jev_train/outputs/selection_$(date +%Y%m%d_%H%M%S)"
setsid --wait "$JEV_RUNNER_PYTHON" -B -u experiments/jev_train/scripts/select_pool.py \
  --pool "$JEV_POOL_ROOT" --config experiments/jev_train/configs/selection/portable_task_reply_v2_38k.json \
  --output "$JEV_SELECTION_OUT" --verify-files
```

配方入口为count、选样seed、领域均匀混合、来源权重指数和正整数回复优先权重。默认38k及 `0.5 / 0.5 / 2` 保留冻结R00参考、整数配额、sqrt和randrange实现，不新增过滤、手工配额或复制采样。调整时在 `jev_train/configs/selection/` 新建配方，传给 `--config` 并提交，选择输出用新目录。

首次下载使用 `--verify-files` 验证完整正文和eval；普通选样只验证注释/参考与正文大小。输出为ID、摘要和两份metadata，执行RESPOND 8%–20%门槛；失败保留摘要且不写ID，不自动调权或补样。

去重seed固定20261015，选样seed只控制池内选择，训练seed控制后续顺序与候选排列。训练可直接使用包内已批准38k ID；复现须保持逐ID正文、seed、world size和accum，双卡不直接改为八卡。`index_cache`不上传，在新机重建。

### 双卡 smoke 与正式训练

`train_pilot.json` 是沿用的通用默认参数表。本页显式覆盖为 2B、已选全部 ID、双卡 `accum=1`；正式训练每 2000 步保存和监控，监控 val 300 条。SwanLab 使用 `DX3906/action_distill`，运行名由本次实验指定；在训练终端完成 SwanLab 登录，或设置 `SWANLAB_API_KEY`。

```bash
: "${JEV_TRAIN_GPUS:?填写已确认的两张GPU，例如0,1}"
: "${JEV_TRAIN_RUN_NAME:?填写本次正式实验的SwanLab运行名}"
JEV_TRAIN_ARGS=(
  --config experiments/jev_train/configs/train_pilot.json
  --model "$JEV_MODEL_ROOT/Mapika-decider-2b"
  --samples "$JEV_POOL_ROOT/pool" --validation_samples "$JEV_POOL_ROOT/eval"
  --train_ids "$JEV_SELECTION_OUT/train_ids.txt" --train_limit 0
  --accum 1 --lr 1e-5 --seed 20261015 --warmup 150
)
JEV_TRAIN_PATH="$JEV_WORKSPACE_ROOT/experiments/jev_train/src:$JEV_WORKSPACE_ROOT/third_party/decider"
JEV_TRAIN_SMOKE_OUT="$JEV_WORKSPACE_ROOT/experiments/jev_train/tmp/${JEV_TRAIN_RUN_NAME}_smoke_$(date +%Y%m%d_%H%M%S)"
setsid --wait env CUDA_VISIBLE_DEVICES="$JEV_TRAIN_GPUS" PYTHONPATH="$JEV_TRAIN_PATH" \
  "$JEV_TRAIN_PYTHON" -B -m torch.distributed.run --standalone --nproc_per_node=2 \
  experiments/jev_train/scripts/train_jev.py "${JEV_TRAIN_ARGS[@]}" \
  --out "$JEV_TRAIN_SMOKE_OUT" --run_name "${JEV_TRAIN_RUN_NAME}_smoke" \
  --smoke --swanlab_mode disabled --max_steps 4 --save_every 2 --eval_every 2 --monitor_val_size 30
```

smoke 在独立 tmp 目录验证实际前反向、监控、检查点和模型导出。检查本次样本与显存记录后，再执行正式训练；两卡协议与恢复说明见[训练 README](../jev_train/README.md#双卡训练)。

```bash
export JEV_TRAIN_OUT="$JEV_WORKSPACE_ROOT/experiments/jev_train/outputs/runs/$JEV_TRAIN_RUN_NAME"
setsid --wait env CUDA_VISIBLE_DEVICES="$JEV_TRAIN_GPUS" PYTHONPATH="$JEV_TRAIN_PATH" \
  "$JEV_TRAIN_PYTHON" -B -m torch.distributed.run --standalone --nproc_per_node=2 \
  experiments/jev_train/scripts/train_jev.py "${JEV_TRAIN_ARGS[@]}" \
  --out "$JEV_TRAIN_OUT" --run_name "$JEV_TRAIN_RUN_NAME" --swanlab_mode cloud \
  --max_steps 0 --eval_every 2000 --save_every 2000 --monitor_val_size 300
```

训练最终权重为 `$JEV_TRAIN_OUT/model/`。冻结模型后做 val 校准与最终 ID/OOD 评测：

```bash
JEV_EVAL_ARGS=(
  --run "$JEV_TRAIN_OUT" --samples "$JEV_POOL_ROOT/eval"
  --majority "$JEV_TRAIN_OUT/training_majority_tools.json"
  --gpus "${JEV_EVAL_GPUS:?填写本次评测GPU}" --seed 20261015
  --limit_val 1000 --limit_test_id 1000 --limit_test_ood 1000
)
export JEV_VAL_OUT="$JEV_WORKSPACE_ROOT/experiments/jev_train/outputs/evals/${JEV_TRAIN_RUN_NAME}_val"
setsid --wait "$JEV_TRAIN_PYTHON" -B -u experiments/jev_train/scripts/eval_jev.py \
  "${JEV_EVAL_ARGS[@]}" --splits val --out "$JEV_VAL_OUT"
setsid --wait "$JEV_TRAIN_PYTHON" -B -u experiments/jev_train/scripts/eval_jev.py \
  "${JEV_EVAL_ARGS[@]}" --splits test_id test_ood --calibration_eval "$JEV_VAL_OUT" \
  --out "$JEV_WORKSPACE_ROOT/experiments/jev_train/outputs/evals/${JEV_TRAIN_RUN_NAME}_test"
```

val 输出 `eval.json` 和 `temperature.json`；测试复用同一模型的 val 温度。Ours 服务使用这份权重与温度，模型冻结后再报告最终测试结果。

## 三、Ours 联合测评

复用公共准备中的 benchmark、执行器服务、API 凭据和解释器。使用第二条路线的最终模型与 val 校准；已有模型时，可直接把下面两个变量设为对应资产的绝对路径。

```bash
cd "$JEV_WORKSPACE_ROOT"
export JEV_OURS_MODEL="${JEV_TRAIN_OUT:?填写本次训练run的绝对路径}/model"
export JEV_OURS_TEMPERATURE="${JEV_VAL_OUT:?填写该模型val评测目录的绝对路径}/temperature.json"
: "${JEV_OURS_GATE:?指定off、G1或G2}"
JEV_HINT_ARGS=() # 默认top-1；自适应top-3时改成下一行
# JEV_HINT_ARGS=(--ours-adaptive-top3-threshold 0.5)
JEV_OURS_PORT=18770
JEV_OURS_URL="http://127.0.0.1:$JEV_OURS_PORT"
"$JEV_RUNNER_PYTHON" -B experiments/migration/services.py --kind jev \
  --python "$JEV_TRAIN_PYTHON" --workspace-root "$JEV_WORKSPACE_ROOT" --model-root "$JEV_MODEL_ROOT" \
  --port "$JEV_OURS_PORT" --gpu "${JEV_OURS_GPU:?指定Jev服务GPU}" \
  --ours-model "$JEV_OURS_MODEL" --ours-temperature "$JEV_OURS_TEMPERATURE"
```

启动打印出的 Jev 服务命令后，从原六条件模板重新生成联合配置。门控 `off` 仍给建议，G1/G2 沿用原规则；通过门控后，自适应 top-3 在 top-1 概率低于 0.5 时展示前三项。smoke 与正式配置复用相同参数。

```bash
JEV_OURS_TAG="${JEV_BENCHMARK}_ours_$(date +%Y%m%d_%H%M%S)"
JEV_OURS_ARGS=(
  --template "experiments/jev_bench/configs/${JEV_BENCHMARK}_main_smoke.json"
  --workspace-root "$JEV_WORKSPACE_ROOT" --model-root "$JEV_MODEL_ROOT"
  --qwen-url "$JEV_QWEN_URL" --actionopd-url "$JEV_ACTIONOPD_URL"
  --star-url "$JEV_STAR_URL" --renderer-url "$JEV_RENDERER_URL" --ours-url "$JEV_OURS_URL"
  --ours-model "$JEV_OURS_MODEL" --ours-temperature "$JEV_OURS_TEMPERATURE"
  --ours-gate "$JEV_OURS_GATE" "${JEV_HINT_ARGS[@]}"
  --conditions native official_jev ours actionopd deepseek_advisor star
)
JEV_OURS_SMOKE_CONFIG="$JEV_WORKSPACE_ROOT/experiments/jev_bench/tmp/${JEV_OURS_TAG}_smoke.json"
"$JEV_RUNNER_PYTHON" -B experiments/migration/deploy.py "${JEV_OURS_ARGS[@]}" \
  --output "$JEV_OURS_SMOKE_CONFIG"
env LITELLM_LOCAL_MODEL_COST_MAP=True "$JEV_BENCH_PYTHON" -B \
  experiments/jev_bench/scripts/run.py --config "$JEV_OURS_SMOKE_CONFIG"
setsid --wait env LITELLM_LOCAL_MODEL_COST_MAP=True \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  "$JEV_BENCH_PYTHON" -B -u experiments/jev_bench/scripts/run.py \
  --config "$JEV_OURS_SMOKE_CONFIG" --run-id "${JEV_OURS_TAG}_smoke" --execute
```

核对建议确实注入、门控/展示符合所选策略、官方评分和成本完整后，执行联合正式批次：

```bash
JEV_OURS_FORMAL_CONFIG="$JEV_WORKSPACE_ROOT/experiments/jev_bench/configs/${JEV_OURS_TAG}.json"
"$JEV_RUNNER_PYTHON" -B experiments/migration/deploy.py "${JEV_OURS_ARGS[@]}" \
  --formal --output "$JEV_OURS_FORMAL_CONFIG"
"$JEV_BENCH_PYTHON" -B experiments/jev_bench/scripts/run.py --config "$JEV_OURS_FORMAL_CONFIG"
git -C "$JEV_WORKSPACE_ROOT/experiments" add "$JEV_OURS_FORMAL_CONFIG"
git -C "$JEV_WORKSPACE_ROOT/experiments" commit --only -m "Configure $JEV_OURS_TAG" -- "$JEV_OURS_FORMAL_CONFIG"
setsid --wait env LITELLM_LOCAL_MODEL_COST_MAP=True \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  "$JEV_BENCH_PYTHON" -B -u experiments/jev_bench/scripts/run.py \
  --config "$JEV_OURS_FORMAL_CONFIG" --run-id "$JEV_OURS_TAG" --execute
```

本批自动生成 Ours 对所选基线的配对比较，结果在 `experiments/jev_bench/outputs/$JEV_OURS_TAG/metrics.json`。已有五基线批次单独保留；当前汇总入口按单批次处理。

PA-Tool 另按[接口命名入口](../jev_bench/README.md#pa-tool先冻结接口再运行独立条件)从零生成全量映射，并用独立 PA 配置完成真实 smoke，核对接口投影、调用还原和官方评分。验收后在联合配置的 `--conditions` 增加 `pa_tool`，同时传 `--pa-mapping <完整mapping.json绝对路径>`，使用新的正式配置和 run-id。预算和映射未确定时，上面的六条件命令可独立使用。

## 结果与恢复

benchmark 的原配置、原 run-id 和源码身份保持一致时，符合恢复条件的 `partial/running/interrupted` 批次在原执行命令末尾加 `--resume`。评分失败先按[评分恢复](../jev_bench/README.md#失败恢复与结果)处理；`--rescore` 可调用 API-Bank 联网工具或 Stable judge。原始失败和评分记录保留。

训练恢复在原训练命令追加 `--resume "$JEV_TRAIN_OUT/checkpoints/step_XXXXXXX"`，保持原输出、运行名、卡数和参数；离线评测重用原命令与输出目录，复用已完成分片。详细条件见[训练恢复](../jev_train/README.md#训练索引指标与恢复)和[离线评测](../jev_train/README.md#独立评测配对与恢复)。

## 目标机验收与正式运行

1. 源码/benchmark清单检查、发布包完整校验与默认配方重放。
2. 环境安装与依赖检查；A100训练前反向/保存恢复及服务实际请求的有界smoke，使用独立tmp输出。
3. 六项全量输入预览、最终模型/温度/门控/映射身份核对；代表性episode按官方评分和成本记录验收。
4. 核对TypeSafe/DeepSeek、Stable的 `TOOLBENCH_KEY`、ToolSandbox订阅/额度及API-Bank联网点；提交最终配置，用户启动正式运行。

执行/恢复/汇总继续使用 `jev_bench/scripts/run.py`、`summarize.py` 和各benchmark原生输入，不新增调度框架或重算历史结果。尚未确认的目标机OS/显存/驱动、完整池发布及许可、最终Ours与PA映射均保留为部署前待办；有CPU测试证据不表示正式训练/测评已经完成。
