# 补充实验推荐运行顺序与完整命令

> 审计日期：2026-09-15；服务器部署路径确认：2026-09-16；续跑／识别验证频率更新：2026-09-18  
> 目标环境：Linux 远程服务器，NVIDIA RTX 4090 24 GB  
> 本文暂不包含 Qwen2.5-VL、GOT-OCR2.0、PaddleOCR-VL 等 LMM/API 实验。
> 下列命令均为远程 Linux Bash 命令，不是在本地 PowerShell 中执行。

## 1. 结论：先跑什么

建议第一个正式完成的实验是 **VSAA counterpart 检测消融**，随后依次运行：

1. VSAA counterpart 六模型受控实验；
2. SVTRv2 + NRTR；
3. OTP λ 敏感性六模型；
4. 补齐其余没有 `best.pth` 的检测/识别对比模型；
5. 重训最终 Ours recognizer，以生成新的 CTC loss、OTP loss、total loss 训练历史；
6. 对全部 16 个识别模型和 13 个检测模型统一运行 validation；
7. 冻结配置与 `best.pth` 后，只运行一次 held-out test；
8. 最后运行四条 E2E OCR pipeline。

把 VSAA counterpart 放在第一位的原因是：当前四个新 counterpart 完全没有权重和结果；该组训练比识别训练短、没有其他模型依赖，适合先验证服务器、数据路径、逐轮验证和 `best.pth` 产出是否正常。与此同时，应尽早导出 E2E 转写模板，因为人工补真实词转写是 E2E 的实际关键路径，但它不占 GPU。

## 2. 当前仓库结果缺口

以下结论来自当前项目目录中的 checkpoint、指标 JSON 和训练历史；服务器若已有额外结果，应先用第 4 节的 dry-run 重新审计。

| 类别 | 当前状态 | 正式处理 |
|---|---|---|
| VSAA counterpart | Strip Pooling、Coordinate Attention、Horizontal-only、Vertical-only 无权重、无 validation/test；DBNet++ 和 VSAA 有旧权重，但没有 counterpart 汇总所需的新格式结果 | 为受控公平性，六个模型全部按同一配置族从头训练 |
| OTP λ sensitivity | λ = 0、0.025、0.05、0.1、0.2、0.5 均无权重及结果 | 六个值全部训练；建议先跑 λ=0.1 检查 OTP 与特殊评价口径，再跑其余值 |
| SVTRv2 + NRTR | 无权重、无 validation/test，现有识别对比汇总也没有该行 | 单独训练并纳入统一 validation/test |
| 旧的四个无权重模型 | SVTRv2、DCM、PP-OCRv5 Det、Hi-SAM 的目录中没有实际 checkpoint；SVTRv2/DCM 现有结果还显示 `num_batches=0` | 结果视为无效，必须训练后重评 |
| CRNN、PARSeq、ABINet | 只有 `last.pth`，没有按每轮 validation 选出的 `best.pth` | 为严格 `best_only` held-out test，重新训练生成 `best.pth` |
| 旧 validation/test JSON | 多数是旧字段格式，缺少统一要求的 `runtime_seconds` 等运行信息 | 所有模型用当前统一 evaluator 重跑 validation/test；无需仅为 CER/CA 公式修改而重训已有有效 `best.pth` |
| 识别训练 loss 图 | 旧的 Ours `metrics.json` 没有 `train_ctc_loss`、`train_ortho_loss`、`train_weighted_ortho_loss`，正 λ 的三条训练曲线无法可靠反推 | 至少重训最终 Ours recognizer；新 λ 实验会自然生成三类 loss |
| held-out test | 当前没有完整的 29 模型新格式统一运行记录 | validation 完成、配置冻结后，用 `best_only` 一次性重跑 |
| E2E OCR | 旧结果只有一条 pipeline，`matched_text_samples=0`，且 IoU=0.5；当前要求的四条 pipeline × validation/test 均无有效结果 | 先补真实 GT word transcription，再生成 8 条正式结果 |

按当前本地 checkpoint 状态，严格 `best_only` 还缺 **18 个 `best.pth`**：12 个识别模型和 6 个检测模型。若把两个已有锚点模型也按 VSAA counterpart 受控协议重训，并把最终 Ours recognizer 为 loss 图重训，建议队列一共是 21 次训练执行。

本地 `data/manifests/*.txt` 是发布用空占位文件；正式服务器实验应使用 `configs/paths/remote_server.yaml` 指向的外部数据清单，不能使用这些空文件。

## 3. 正式实验必须遵守的顺序

- 检测训练每轮 validation；全部识别实验改为每 50 轮 validation，并在最后一轮验证。由各自 YAML 的 `best_metric` 产生 `best.pth`。
- 单独 validation 与 held-out test 使用同一 evaluator 和同一模型配置，只改变 manifest split。
- 所有识别模型统一使用严格 WA、`CA=(N-S-D)/N` 和 `CER=(S+D+I)/N`，不再使用逐样本容错或字符位置例外。
- 所有检测模型在原始 validation/test 图像上统一使用 `IoU=0.75`，不再启用评估退化。
- 每组的第一段命令仅用于**首次从头训练**；中断后使用该组下方的**断点续跑命令**，显式指定本轮的 `last.pth`。不要重复执行首次训练命令，否则会从第 1 轮开始，并可能覆盖已有权重。
- 只能恢复本轮相同代码、配置、split 和训练参数产生的 checkpoint，不能误接旧协议或 smoke test 权重。文中的续训命令使用 `--resume`；如果文件缺失或无法读取，会报错停止，而不是自动从头训练。
- 在所有 validation 结果确认、配置和 checkpoint 哈希冻结以前，不运行 test。
- 不根据 test 结果修改 λ、阈值、解码方式或模型选择。

### 3.1 识别验证频率更新：每 50 轮一次

全部 16 个识别配置（主模型、对比、原消融及 OTP λ 消融）设置为
`train.val_interval: 50`。ABINet、DCM、PARSeq、SVTRv2、SVTRv2+NRTR 同时设置
`train.rerank_val_interval: 50`；其余模型不新增 rerank。200 轮正式训练只在
第 50、100、150、200 轮验证，不额外强制第 1 轮验证；最后一轮即使不是
50 的倍数也验证，debug/smoke test 不受限制。统一严格指标、解码规则、
loss 和 batch size 在同一对比组内保持不变。

同步到服务器的文件不能只有 YAML，还需要：

- `configs/recognition/` 下的全部 16 个 YAML；
- `configs/experiments/rec_otp_lambda_sensitivity.yaml`；
- `scripts/train_recognition.py`（取消强制第 1 轮验证，避免 rerank best 指标缺失）；
- `src/manchu_ocr/utils/recognition_curve_data.py`；
- `tools/plot_recognition_comparison_curves.py`、`tools/plot_recognition_formal_curves.py`、`tools/plot_final_recognition_model.py`；
- `tests/test_validation_policy.py`、`tests/test_training_resume.py`、`tests/test_otp_lambda_sensitivity.py`、`tests/test_recognition_curve_data.py`；
- 本文以及 `docs/current_detection_recognition_validation_logic.md`、`docs/ablation_plan.md`。

训练进程只在启动时读取配置，上传文件不会改变正在运行的验证频率。
如需让当前 SVTRv2+NRTR 改用新频率，建议等当前轮验证完成并确认 `last.pth`
更新后，再结束旧进程、同步文件，并用顺序 2 的 `--resume last.pth` 命令续训；
不要同时启动第二个训练进程。仅验证频率的本次明确变更可恢复已有正式断点，
其他训练参数必须保持不变；保留既有历史及旧 best，再与后续验证结果比较，
不会自动清空旧权重或旧指标。

更新后的 `last.pth` 仍每轮保存，续训不必等到第 50 轮。从头训练通常在
第 50 轮才首次生成 `best.pth`；它只能从实测验证轮选择，因此可能错过两次
验证之间的最佳 epoch。CTC、OTP、total loss 仍每轮记录和绘图；validation
loss、WA、CA、CER 在 200 轮中只有 4 个实测点。绘图跳过未验证轮次的 NaN，
在真实 epoch 标记实测点，连线不代表中间轮次有验证结果。
独立 validation、held-out test 和 E2E 的运行逻辑不因训练验证间隔改变。

同步后可在服务器项目根目录验证：

```bash
cd /root/code/OCR_Manchu
source scripts/set_remote_paths.sh
export PYTHONPATH="${OCR_MANCHU_PROJECT_ROOT}/src:${PYTHONPATH:-}"
python3 -m pytest -q \
  tests/test_validation_policy.py \
  tests/test_training_resume.py \
  tests/test_otp_lambda_sensitivity.py \
  tests/test_recognition_curve_data.py
```

## 4. 服务器初始化、审计与 smoke test

### 4.1 进入一个可保持连接的会话

```bash
tmux new -s manchu_ocr
```

如果已经在 tmux 中，可跳过该命令。退出 VS/SSH 后，先检查原会话：

```bash
tmux ls
```

原会话仍在时使用：

```bash
tmux attach -t manchu_ocr
```

先查看会话内任务是否仍在运行；若只是客户端断线，不需要重新执行训练命令。
可在另一个终端只读检查：

```bash
pgrep -af '[p]ython.*(train_detection|train_recognition|eval_detection|eval_recognition|eval_full_ocr)\.py' || true
```

同一模型不得同时启动两个训练进程，以免共同写入同一个 `best.pth`、
`last.pth` 和指标目录。确认原进程已经退出后，才执行下文恢复命令。
tmux 能维持终端断线后的会话，但不能防止服务器重启、平台停机或系统终止任务。

### 4.2 设置路径与运行环境

以下三个路径已确认是实际部署目录。先同步本轮新增的
`scripts/set_remote_paths.sh`、`tools/migrate_remote_paths.py`、
`tests/test_remote_path_migration.py` 和 `tests/test_training_resume.py` 到服务器。
所有命令在远程 Bash 会话中执行。

```bash
set -euo pipefail

cd /root/code/OCR_Manchu
source scripts/set_remote_paths.sh
export PYTHON=python3
export PYTHON_BIN=python3
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${OCR_MANCHU_PROJECT_ROOT}/src:${PYTHONPATH:-}"

cd "${OCR_MANCHU_PROJECT_ROOT}"
mkdir -p "${OCR_MANCHU_OUTPUT_ROOT}/run_logs" "${OCR_MANCHU_OUTPUT_ROOT}/audit"
```

若服务器环境尚未安装项目依赖：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

确认当前 Python 实际加载的是带 CUDA 的 PyTorch：

```bash
nvidia-smi
python3 - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_runtime:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available(), "当前 Python 环境没有可用 CUDA，停止正式实验"
PY
```

### 4.3 只审计，不重新划分数据

上传的数据清单仍可能包含旧 Windows 绝对路径，例如
`C:/Users/ahs/Desktop/Manchu_Recognition_Data/...`。YAML 的根目录正确
并不会自动修正这些清单内部路径。首次训练前先预览迁移：

```bash
python3 tools/migrate_remote_paths.py \
  --config configs/paths/remote_server.yaml \
  --dry-run
```

确认路径和目标文件检查通过后应用：

```bash
python3 tools/migrate_remote_paths.py \
  --config configs/paths/remote_server.yaml \
  --apply
```

工具只修改 processed 清单、CSV、标注/元信息中的路径字段，以及项目内
存在的旧 full-OCR 清单；不会重新划分 split、调整样本顺序、转写、
检测框、charset、transition matrix 或实验指标。每个修改文件旁都会保留
`.paths_backup_<UTC>` 原始备份；重复执行不会再次改动已经正确的路径。
若目标图像/标注未上传完整，工具会在写入前停止，不能用重新划分数据绕过。

本轮对本地数据的只读迁移预览得到：检测 train/val/test 为
400/50/50，识别为 164974/20621/20623；所有识别 label 及其顺序均未改变。
服务器迁移日志应保留相同样本数。此预览不是服务器文件存在性检查。

已有正式 split 时，不要为本轮实验重新生成 train/val/test。迁移后检查路径、数据、配置和模型构建：

```bash
python3 tools/preflight_remote.py \
  --config configs/paths/remote_server.yaml \
  --stage all \
  2>&1 | tee "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/00_preflight.log"
```

冻结当前数据清单哈希：

```bash
sha256sum \
  "${OCR_MANCHU_DET_ROOT}/processed/train.txt" \
  "${OCR_MANCHU_DET_ROOT}/processed/val.txt" \
  "${OCR_MANCHU_DET_ROOT}/processed/test.txt" \
  "${OCR_MANCHU_REC_ROOT}/processed/train.txt" \
  "${OCR_MANCHU_REC_ROOT}/processed/val.txt" \
  "${OCR_MANCHU_REC_ROOT}/processed/test.txt" \
  | tee "${OCR_MANCHU_OUTPUT_ROOT}/audit/manifests.sha256"
```

检查本轮新增代码；这里明确不运行 LMM 测试：

```bash
python3 -m pytest -q \
  tests/test_remote_path_migration.py \
  tests/test_training_resume.py \
  tests/test_directional_attention_counterparts.py \
  tests/test_vsaa.py \
  tests/test_svtrv2_nrtr.py \
  tests/test_otp_lambda_sensitivity.py \
  tests/test_recognition_curve_data.py \
  tests/test_recognition_metrics.py \
  tests/test_validation_policy.py \
  tests/test_experiment_result.py \
  tests/test_heldout_runner.py \
  tests/test_e2e_ocr.py
```

### 4.4 在独立目录做三项最小 smoke test

smoke test 不写入正式 `outputs`，用于尽早发现新模块、显存或数据读取问题：

```bash
export OCR_MANCHU_SMOKE_ROOT=/root/code/OCR_Manchu/outputs_smoke

OCR_MANCHU_OUTPUT_ROOT="${OCR_MANCHU_SMOKE_ROOT}" \
python3 scripts/train_detection.py \
  --config configs/detection/dbnetpp_strip_pooling.yaml \
  --epochs 1 --max-train-batches 2 --max-val-batches 2 \
  --batch-size 2 --num-workers 2

OCR_MANCHU_OUTPUT_ROOT="${OCR_MANCHU_SMOKE_ROOT}" \
python3 scripts/train_recognition.py \
  --config configs/recognition/svtrv2_nrtr_baseline.yaml \
  --epochs 1 --max-train-batches 2 --max-val-batches 2 \
  --batch-size 4 --num-workers 2

OCR_MANCHU_OUTPUT_ROOT="${OCR_MANCHU_SMOKE_ROOT}" \
python3 scripts/train_recognition.py \
  --config configs/recognition/svtr_dab_otp_lambda_0p1.yaml \
  --epochs 1 --max-train-batches 2 --max-val-batches 2 \
  --batch-size 4 --num-workers 2
```

三项均成功且各自生成 `best.pth` 后，再开始正式队列。

## 5. 同时启动 E2E 转写准备（不占用正式 GPU 队列）

先导出 validation/test 页的 GT word review crops 和转写表：

```bash
python3 tools/prepare_e2e_transcriptions.py export \
  --config configs/paths/remote_server.yaml \
  --splits val test \
  --save-review-crops \
  2>&1 | tee "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/00_e2e_transcription_export.log"
```

随后人工填写：

```text
${OCR_MANCHU_DET_ROOT}/processed/e2e/transcriptions.csv
```

每个词必须填写真实 Romanized Manchu transcription，并在复核后将 `status` 标为 `done`。review crops 只用于标注，不会成为正式 E2E recognizer 的输入；正式 E2E 仍只裁 detector-predicted boxes。

## 6. 正式训练队列

所有命令都从项目根目录执行，并沿用第 4.2 节环境变量。

### 6.0 断点能力、恢复前检查与共同设置

已核对本节的 **21 个正式训练模型**：检测模型统一调用
`scripts/train_detection.py`，识别模型统一调用
`scripts/train_recognition.py`，两者均支持 `--resume` 和 `--auto-resume`。

| 队列 | 训练模型数 | 脚本 | 保持的 batch size | num_workers |
|---|---:|---|---|---:|
| 顺序 1：VSAA counterpart | 6 | train_detection.py | 16 | 8 |
| 顺序 2：SVTRv2 + NRTR | 1 | train_recognition.py | 64 | 8 |
| 顺序 3：OTP λ sensitivity | 6 | train_recognition.py | 96 | 8 |
| 顺序 4：其余识别对比模型 | 5 | train_recognition.py | CRNN=128，其余=64 | 8 |
| 顺序 5：其余检测对比模型 | 2 | train_detection.py | 16 | 8 |
| 顺序 6：最终 Ours recognizer | 1 | train_recognition.py | 96 | 8 |

恢复能力和限制：

- 每轮训练结束都保存 `last.pth`；若该轮需要 validation，则在验证完成、更新 best 状态后保存。从其中记录的 `epoch + 1` 继续。中断的那一轮需要重跑，不是 batch 级恢复。
- 两个脚本恢复模型、optimizer、scheduler、AMP scaler 及 best 指标；识别脚本还恢复原始训练权重 `train_model` 和启用时的 EMA 权重。完整历史保存在相应 `metrics.json`，恢复时保留不超过断点 epoch 的记录。
- `best.pth` 用于验证选模和最终评估，不代表训练已经完成，也不应代替最新的 `last.pth` 续训。
- `--auto-resume` 在找不到 `last.pth` 时会从头开始；因此下面全部使用显式 `--resume`，避免意外从第 1 轮开始。
- 第一轮保存前中断，没有可恢复的断点，只能从第 1 轮开始。当前未保存随机数/数据加载状态，续训不能保证与从未中断的训练逐步完全一致；checkpoint 写入时被强制终止也可能造成文件损坏。
- 除第 3.1 节明确授权的识别验证频率变更外，恢复时不改变 YAML、代码、batch size、梯度累积、总 epoch、split、charset 或 transition matrix。旧格式识别 checkpoint 若缺少 `train_model`，脚本会警告；不要把它当成本轮完整断点。
- 训练和评估没有按时间自动暂停、自动重启的功能；日志空窗可能是无逐批日志的 validation，不能仅凭日志不更新就重启任务。

**每次重新进入 VS/SSH 后**，先激活与原运行相同的 CUDA/PyTorch 环境，
进入或重新建立 tmux 会话，并在该会话内重新执行以下共同设置。
这一步不训练模型，也不会重新划分数据：

```bash
set -euo pipefail
cd /root/code/OCR_Manchu
source scripts/set_remote_paths.sh
export PYTHON=python3
export PYTHON_BIN=python3
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${OCR_MANCHU_OUTPUT_ROOT}/run_logs" "${OCR_MANCHU_OUTPUT_ROOT}/audit"
```

可先只读检查本轮已生成断点的 epoch（不会加载到 GPU，不会改写权重）：

```bash
python3 - <<'PY'
import os
from pathlib import Path

import torch

root = Path(os.environ["OCR_MANCHU_OUTPUT_ROOT"]) / "checkpoints"
for task in ("detection", "recognition"):
    for path in sorted((root / task).glob("*/last.pth")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        required = {"epoch", "model", "optimizer", "scheduler", "scaler"}
        missing = sorted(required.difference(checkpoint))
        if task == "recognition" and "train_model" not in checkpoint:
            missing.append("train_model")
        if missing:
            raise RuntimeError(f"Incomplete/older checkpoint {path}: {missing}")
        completed = int(checkpoint["epoch"])
        print(f"{task}/{path.parent.name}: completed_epoch={completed}, next_epoch={completed + 1}, checkpoint={path}")
        del checkpoint
PY
```

该检查只验证文件可读、关键字段和 epoch，不能证明其训练协议与本轮相同；
仍需对照保存的 config、运行日志和数据清单哈希确认。
如果断点曾被误启动的第 1 轮覆盖，命令不能凭空找回更早的训练进度。

各组下方的循环列出了完整模型名单，方便逐项恢复。**只恢复一个中断模型时，
把 `for name in ...` 的名单只保留该模型**；尚未启动的模型用该组首次训练命令，
同样只保留未启动名单，不要重新训练已经完成的项目。
某个模型没有 `last.pth` 时，续跑命令会停止，不会静默从头开始。
已达到目标总 epoch 且存在 `best.pth` 的同协议断点，训练脚本会正常跳过训练。

下面的恢复日志均用 `tee -a` 追加，保留中断前的运行记录。启动后必须看到：

```text
Resumed from: .../last.pth, completed_epoch=k, start_epoch=k+1, ...
```

例如断点 epoch 为 2，应从第 3 轮开始；没有上述记录就不能认为已成功续训。

### 顺序 1：VSAA counterpart 六模型

即使 DBNet++ 与 VSAA 已有旧权重，也建议在该受控组中一起从头训练，以保证 backbone、split、epoch、optimizer、input size 和训练代码版本一致。

```bash
for name in \
  dbnetpp_official_baseline \
  dbnetpp_strip_pooling \
  dbnetpp_coordinate_attention \
  dbnetpp_horizontal_strip \
  dbnetpp_vertical_strip \
  dbnetpp_vsaa
do
  python3 scripts/train_detection.py \
    --config "configs/detection/${name}.yaml" \
    --batch-size 16 \
    --num-workers 8 \
    2>&1 | tee "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/01_${name}.log"

  test -f "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/detection/${name}/best.pth"
done
```

#### 顺序 1 的断点续跑命令

先完成第 6.0 节共同设置，再仅保留需要恢复的模型执行：

```bash
for name in \
  dbnetpp_official_baseline \
  dbnetpp_strip_pooling \
  dbnetpp_coordinate_attention \
  dbnetpp_horizontal_strip \
  dbnetpp_vertical_strip \
  dbnetpp_vsaa
do
  python3 -u scripts/train_detection.py \
    --config "configs/detection/${name}.yaml" \
    --batch-size 16 \
    --num-workers 8 \
    --resume "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/detection/${name}/last.pth" \
    2>&1 | tee -a "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/01_${name}.log"

  test -f "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/detection/${name}/best.pth"
done
```

### 顺序 2：SVTRv2 + NRTR

这是新增识别架构，先单独跑完，可尽早发现 NRTR 辅助训练分支的问题；validation/test 仍使用与 SVTRv2 相同的 CTC 推理与指标逻辑。

```bash
python3 scripts/train_recognition.py \
  --config configs/recognition/svtrv2_nrtr_baseline.yaml \
  --batch-size 64 \
  --num-workers 8 \
  2>&1 | tee "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/02_svtrv2_nrtr_baseline.log"

test -f "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/recognition/svtrv2_nrtr_baseline/best.pth"
```

#### 顺序 2 的断点续跑命令

先完成第 6.0 节共同设置；以下保持 batch size=64，不重新训练第 1 轮：

```bash
python3 -u scripts/train_recognition.py \
  --config configs/recognition/svtrv2_nrtr_baseline.yaml \
  --batch-size 64 \
  --num-workers 8 \
  --resume "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/recognition/svtrv2_nrtr_baseline/last.pth" \
  2>&1 | tee -a "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/02_svtrv2_nrtr_baseline.log"

test -f "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/recognition/svtrv2_nrtr_baseline/best.pth"
```

### 顺序 3：OTP λ 敏感性

建议先跑 λ=0.1，因为它同时检查正 OTP loss、三条训练 loss 记录和统一严格 validation 逻辑；之后跑 λ=0 控制组，再补其余四个取值。训练顺序不改变每个配置内固定的 seed。

```bash
for name in \
  svtr_dab_otp_lambda_0p1 \
  svtr_dab_otp_lambda_0 \
  svtr_dab_otp_lambda_0p025 \
  svtr_dab_otp_lambda_0p05 \
  svtr_dab_otp_lambda_0p2 \
  svtr_dab_otp_lambda_0p5
do
  python3 scripts/train_recognition.py \
    --config "configs/recognition/${name}.yaml" \
    --batch-size 96 \
    --num-workers 8 \
    2>&1 | tee "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/03_${name}.log"

  test -f "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/recognition/${name}/best.pth"
done
```

这里不使用 `scripts/run_rec_otp_lambda_sensitivity.sh`，因为该一键脚本在训练结束后会立即访问 test；正式流程要等全部 validation 结束并冻结配置后再访问 test。

#### 顺序 3 的断点续跑命令

先完成第 6.0 节共同设置，再仅保留需要恢复的 λ 模型执行：

```bash
for name in \
  svtr_dab_otp_lambda_0p1 \
  svtr_dab_otp_lambda_0 \
  svtr_dab_otp_lambda_0p025 \
  svtr_dab_otp_lambda_0p05 \
  svtr_dab_otp_lambda_0p2 \
  svtr_dab_otp_lambda_0p5
do
  python3 -u scripts/train_recognition.py \
    --config "configs/recognition/${name}.yaml" \
    --batch-size 96 \
    --num-workers 8 \
    --resume "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/recognition/${name}/last.pth" \
    2>&1 | tee -a "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/03_${name}.log"

  test -f "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/recognition/${name}/best.pth"
done
```

### 顺序 4：补齐识别对比模型的 `best.pth`

SVTRv2 和 DCM 当前无实际权重；CRNN、PARSeq、ABINet 当前只有旧 `last.pth`。下列命令全部重新训练，SVTRv2+NRTR 已在顺序 2 完成，不重复运行。

```bash
for name in \
  svtrv2_baseline \
  dcm_baseline \
  crnn_baseline \
  parseq_baseline \
  abinet_baseline
do
  case "${name}" in
    crnn_baseline) batch_size=128 ;;
    *) batch_size=64 ;;
  esac

  python3 scripts/train_recognition.py \
    --config "configs/recognition/${name}.yaml" \
    --batch-size "${batch_size}" \
    --num-workers 8 \
    2>&1 | tee "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/04_${name}.log"

  test -f "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/recognition/${name}/best.pth"
done
```

#### 顺序 4 的断点续跑命令

先完成第 6.0 节共同设置，再仅保留需要恢复的识别模型执行：

```bash
for name in \
  svtrv2_baseline \
  dcm_baseline \
  crnn_baseline \
  parseq_baseline \
  abinet_baseline
do
  case "${name}" in
    crnn_baseline) batch_size=128 ;;
    *) batch_size=64 ;;
  esac

  python3 -u scripts/train_recognition.py \
    --config "configs/recognition/${name}.yaml" \
    --batch-size "${batch_size}" \
    --num-workers 8 \
    --resume "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/recognition/${name}/last.pth" \
    2>&1 | tee -a "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/04_${name}.log"

  test -f "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/recognition/${name}/best.pth"
done
```

### 顺序 5：补齐检测对比模型

```bash
for name in ppocrv5_det_baseline hisam_baseline
do
  python3 scripts/train_detection.py \
    --config "configs/detection/${name}.yaml" \
    --batch-size 16 \
    --num-workers 8 \
    2>&1 | tee "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/05_${name}.log"

  test -f "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/detection/${name}/best.pth"
done
```

#### 顺序 5 的断点续跑命令

先完成第 6.0 节共同设置，再仅保留需要恢复的检测模型执行：

```bash
for name in ppocrv5_det_baseline hisam_baseline
do
  python3 -u scripts/train_detection.py \
    --config "configs/detection/${name}.yaml" \
    --batch-size 16 \
    --num-workers 8 \
    --resume "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/detection/${name}/last.pth" \
    2>&1 | tee -a "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/05_${name}.log"

  test -f "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/detection/${name}/best.pth"
done
```

### 顺序 6：重训最终 Ours recognizer 以生成新 loss 图

标准 CER/CA 的重新计算本身只需要重评，不要求重训；但旧训练历史不能忠实恢复 CTC loss、OTP loss、total loss 三条训练曲线，因此该模型需重训一次。

```bash
python3 scripts/train_recognition.py \
  --config configs/recognition/svtr_official_dab_lortho.yaml \
  --batch-size 96 \
  --num-workers 8 \
  2>&1 | tee "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/06_svtr_official_dab_lortho.log"

test -f "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/recognition/svtr_official_dab_lortho/best.pth"
```

#### 顺序 6 的断点续跑命令

先完成第 6.0 节共同设置，再恢复本轮最终 Ours recognizer：

```bash
python3 -u scripts/train_recognition.py \
  --config configs/recognition/svtr_official_dab_lortho.yaml \
  --batch-size 96 \
  --num-workers 8 \
  --resume "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/recognition/svtr_official_dab_lortho/last.pth" \
  2>&1 | tee -a "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/06_svtr_official_dab_lortho.log"

test -f "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints/recognition/svtr_official_dab_lortho/best.pth"
```

这不是读取旧 Ours 权重来补齐历史 loss：只有本轮已经记录 CTC/OTP/total
loss 的正式训练中断点才能恢复。中断那一轮重新计算 loss；此前已完成轮次的
历史会从 `metrics.json` 读取保留。

## 7. 统一 validation：先验证全部 29 个模型

训练结束后先做严格 dry-run。输出必须显示 `models_ready=29/29`，否则不要开始 validation，更不要开始 test。

```bash
python3 scripts/run_heldout_tests.py \
  --plan configs/experiments/heldout_all_models.yaml \
  --tasks all \
  --splits validation \
  --python-bin python3 \
  --checkpoint-policy best_only \
  --recognition-batch-size 192 \
  --detection-batch-size 8 \
  --num-workers 8 \
  --dry-run
```

dry-run 通过后正式运行 validation：

```bash
python3 scripts/run_heldout_tests.py \
  --plan configs/experiments/heldout_all_models.yaml \
  --tasks all \
  --splits validation \
  --python-bin python3 \
  --checkpoint-policy best_only \
  --recognition-batch-size 192 \
  --detection-batch-size 8 \
  --num-workers 8 \
  --fail-fast \
  2>&1 | tee "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/07_all_models_validation.log"
```

数量检查：

```bash
test "$(find "${OCR_MANCHU_OUTPUT_ROOT}/metrics/recognition" -mindepth 2 -maxdepth 2 -name eval_val.json | wc -l)" -eq 16
test "$(find "${OCR_MANCHU_OUTPUT_ROOT}/metrics/detection" -mindepth 2 -maxdepth 2 -name eval_val.json | wc -l)" -eq 13
```

此时查看 validation，完成 λ 解释、错误分析和模型选择。不要打开或据此反复运行 test。

### 7.1 validation 中断后的队列续跑

统一 evaluator/runner **没有 batch 级恢复或自动跳过已有结果的功能**。
已完成的模型不需要重新训练，也不需要重复评估；中断的模型需要从该
validation split 的第一个样本重新评估。仅恢复中断项和未完成项即可。

完成判定以本轮 `outputs/metrics/experiment_runs/heldout_all_models_<UTC>.json`
中的 `status: completed` 及其对应的新格式结果为依据；日志中的 `[COMMAND]`
只是预先打印的计划，不代表已执行完成。不能只看 JSON 文件是否存在，
因为它可能是本轮开始前留下的旧结果。若进程在第一个模型完成前退出，
可能尚未生成本轮 summary，该模型应重新评估。

先完成第 6.0 节共同设置。下面示例只补 SVTRv2+NRTR 的 validation；
将 `models` 数组改为实际中断及未完成模型的**实验名**（YAML 中的
`experiment.name`），可以同时列出多个识别/检测模型，支持计划内全部 29 个模型：

```bash
models=(svtrv2_nrtr_baseline)

python3 -u scripts/run_heldout_tests.py \
  --plan configs/experiments/heldout_all_models.yaml \
  --tasks all \
  --splits validation \
  --models "${models[@]}" \
  --python-bin python3 \
  --output-root "${OCR_MANCHU_OUTPUT_ROOT}" \
  --checkpoint-policy best_only \
  --recognition-batch-size 192 \
  --detection-batch-size 8 \
  --num-workers 8 \
  --fail-fast \
  2>&1 | tee -a "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/07_all_models_validation.log"
```

保留原配置和 checkpoint，不添加解码或指标覆盖参数；评估器固定执行无退化、IoU 0.75 与严格识别指标。
补齐后重新执行本节数量检查，并核对所有结果的来源；文件数量正确不等于
已经确认本轮协议一致。一次局部续跑生成的 run summary 只列此次选中的模型，
需要和本轮中断前的 summary 一并保留作为运行记录。

## 8. 冻结配置和 checkpoint

在首次正式 test 前记录哈希：

```bash
find "${OCR_MANCHU_OUTPUT_ROOT}/checkpoints" \
  -type f -name best.pth -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${OCR_MANCHU_OUTPUT_ROOT}/audit/best_checkpoints_before_test.sha256"

find configs/detection configs/recognition configs/experiments \
  -type f -name '*.yaml' -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${OCR_MANCHU_OUTPUT_ROOT}/audit/configs_before_test.sha256"
```

建议把 validation 决策写入一个只读实验日志后，再进入下一节。

如果冻结命令中断且 test 尚未开始，可以重新执行本节哈希生成命令。
test 已经开始后，不要重新生成基准哈希来掩盖文件变化；恢复 test/E2E 前
用第 9.1 节的 `sha256sum -c` 检查冻结文件是否保持不变。

## 9. 一次性运行全部 held-out test

先做 test 命令 dry-run；这一步只打印命令，不执行推理：

```bash
PYTHON=python3 \
CHECKPOINT_POLICY=best_only \
REC_EVAL_BATCH_SIZE=192 \
DET_EVAL_BATCH_SIZE=8 \
EVAL_NUM_WORKERS=8 \
HELDOUT_DRY_RUN=1 \
bash scripts/run_all_heldout_tests.sh
```

确认 29 个模型、路径和 `best.pth` 均正确后，正式运行一次：

```bash
PYTHON=python3 \
CHECKPOINT_POLICY=best_only \
REC_EVAL_BATCH_SIZE=192 \
DET_EVAL_BATCH_SIZE=8 \
EVAL_NUM_WORKERS=8 \
bash scripts/run_all_heldout_tests.sh \
  2>&1 | tee "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/09_all_models_heldout_test.log"
```

数量检查：

```bash
test "$(find "${OCR_MANCHU_OUTPUT_ROOT}/metrics/recognition" -mindepth 2 -maxdepth 2 -name eval_test.json | wc -l)" -eq 16
test "$(find "${OCR_MANCHU_OUTPUT_ROOT}/metrics/detection" -mindepth 2 -maxdepth 2 -name eval_test.json | wc -l)" -eq 13
ls -lt "${OCR_MANCHU_OUTPUT_ROOT}/metrics/experiment_runs" | head
```

### 9.1 held-out test 中断后的队列续跑

这属于同一次冻结协议的未完成测试补跑，不进行任何重新选模或参数调整。
恢复粒度是**模型 + split**，不是 batch；已完成项保留，中断项重新评估
完整 test split。完成判定和旧结果识别规则与第 7.1 节相同。

先完成第 6.0 节共同设置，再检查冻结文件；哈希不一致时停止并查明原因，
不要继续混用结果，也不要重写基准哈希：

```bash
sha256sum -c "${OCR_MANCHU_OUTPUT_ROOT}/audit/best_checkpoints_before_test.sha256"
sha256sum -c "${OCR_MANCHU_OUTPUT_ROOT}/audit/configs_before_test.sha256"
sha256sum -c "${OCR_MANCHU_OUTPUT_ROOT}/audit/manifests.sha256"
```

下面示例只补 SVTRv2+NRTR 的 test。将 `models` 数组改为本轮实际中断及
未完成的模型名单，可以同时包含识别和检测模型。使用 Python runner 的
`--models` 过滤，不再次运行无过滤的一键 test 脚本，以免重跑已完成项：

```bash
models=(svtrv2_nrtr_baseline)

python3 -u scripts/run_heldout_tests.py \
  --plan configs/experiments/heldout_all_models.yaml \
  --tasks all \
  --splits test \
  --models "${models[@]}" \
  --python-bin python3 \
  --output-root "${OCR_MANCHU_OUTPUT_ROOT}" \
  --checkpoint-policy best_only \
  --recognition-batch-size 192 \
  --detection-batch-size 8 \
  --num-workers 8 \
  --fail-fast \
  2>&1 | tee -a "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/09_all_models_heldout_test.log"
```

补齐后执行本节数量检查，按第 10 节重新生成汇总；这些操作不重新训练模型。

## 10. 生成单任务汇总与训练曲线

统一 runner 生成的是 `eval_val.json` 和 `eval_test.json`，检测均已按无退化、IoU 0.75 的严格协议运行。因此汇总 VSAA counterpart 时显式传空 suffix，不再重复推理。

```bash
python3 tools/summarize_vsaa_counterparts.py \
  --root "${OCR_MANCHU_OUTPUT_ROOT}/metrics/detection" \
  --suffix "" \
  --output-prefix detection_vsaa_counterparts

python3 tools/summarize_otp_lambda_sensitivity.py \
  --root "${OCR_MANCHU_OUTPUT_ROOT}/metrics/recognition"

python3 tools/summarize_recognition_comparison.py \
  --root "${OCR_MANCHU_OUTPUT_ROOT}/metrics/recognition"

python3 tools/summarize_recognition_formal.py \
  --root "${OCR_MANCHU_OUTPUT_ROOT}/metrics/recognition"

python3 tools/summarize_detection_comparison.py \
  --detection-root "${OCR_MANCHU_OUTPUT_ROOT}/metrics/detection" \
  --baseline-root "${OCR_MANCHU_OUTPUT_ROOT}/metrics/baselines" \
  --suffix "" \
  --output-prefix detection_comparison_strict

python3 tools/summarize_detection_formal.py \
  --root "${OCR_MANCHU_OUTPUT_ROOT}/metrics/detection" \
  --suffix "" \
  --output-prefix detection_ablation_strict

python3 tools/plot_recognition_formal_curves.py \
  --root "${OCR_MANCHU_OUTPUT_ROOT}/metrics/recognition" \
  --out-dir "${OCR_MANCHU_OUTPUT_ROOT}/visualizations/paper_figures"

python3 tools/plot_recognition_comparison_curves.py \
  --root "${OCR_MANCHU_OUTPUT_ROOT}/metrics/recognition" \
  --out-dir "${OCR_MANCHU_OUTPUT_ROOT}/visualizations/paper_figures"

python3 tools/plot_final_recognition_model.py \
  --root "${OCR_MANCHU_OUTPUT_ROOT}/metrics/recognition" \
  --out-dir "${OCR_MANCHU_OUTPUT_ROOT}/visualizations/paper_figures/main_models" \
  --exp-name svtr_official_dab_lortho \
  --label "SVTR + DAB + OTP"

python3 tools/plot_detection_formal_curves.py \
  --root "${OCR_MANCHU_OUTPUT_ROOT}/metrics/detection" \
  --out-dir "${OCR_MANCHU_OUTPUT_ROOT}/visualizations/paper_figures"
```

重点输出：

```text
outputs/metrics/detection/detection_vsaa_counterparts.{json,csv,md}
outputs/metrics/recognition/recognition_otp_lambda_sensitivity.{json,csv,md}
outputs/metrics/recognition/recognition_comparison.{json,md}
outputs/metrics/experiment_runs/heldout_all_models_<UTC>.json
outputs/visualizations/paper_figures/
```

本节不涉及训练断点。汇总或绘图中断后，可在确认输入结果完整的前提下
重新执行本节对应命令（或整段命令）；它们读取已有结果生成汇总/图像，
不会重新训练或推理。E2E 标注准备中断时重新运行 `audit` 检查完整性，
不要覆盖已经人工填写的转写表。

## 11. 最后运行四条 E2E OCR pipeline

人工转写完成后应用并审计；任何缺词都会使正式 E2E 停止：

```bash
python3 tools/prepare_e2e_transcriptions.py apply \
  --config configs/paths/remote_server.yaml \
  --splits val test

python3 tools/prepare_e2e_transcriptions.py audit \
  --config configs/paths/remote_server.yaml \
  --splits val test
```

先 dry-run，必须看到 4/4 pipelines、validation/test 两个有效 manifest，且没有 generic `text` 占位标签：

```bash
python3 scripts/run_e2e_ocr_experiments.py \
  --plan configs/experiments/e2e_ocr.yaml \
  --splits validation test \
  --python-bin python3 \
  --output-root "${OCR_MANCHU_OUTPUT_ROOT}" \
  --device cuda \
  --checkpoint-policy best_only \
  --dry-run
```

因为四条 pipeline、统一严格指标和 checkpoint 已在前面冻结，正式命令对每条 pipeline 依次运行 validation 和 test，共完成 8 次评估：

```bash
python3 scripts/run_e2e_ocr_experiments.py \
  --plan configs/experiments/e2e_ocr.yaml \
  --splits validation test \
  --python-bin python3 \
  --output-root "${OCR_MANCHU_OUTPUT_ROOT}" \
  --device cuda \
  --checkpoint-policy best_only \
  --fail-fast \
  2>&1 | tee "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/11_e2e_four_pipelines.log"
```

最终 E2E 汇总：

```text
outputs/metrics/e2e_ocr/e2e_comparison_latest.json
outputs/metrics/e2e_ocr/e2e_comparison_latest.csv
```

该 CSV 应包含 4 条 pipeline × 2 个 split = 8 行，并记录 detector/recognizer checkpoint、manifest、运行时间、E2E WA、E2E CER、page-level CER/score、missed words 和 false positives。

### 11.1 E2E 中断后的队列续跑

E2E evaluator **没有 page/batch 级断点恢复或自动跳过功能**。
恢复粒度是 **pipeline + split**：已完成组合保留；中断组合从该 split
第一页重新评估，不重新训练 detector/recognizer。
按本轮 `outputs/metrics/e2e_ocr/e2e_run_<UTC>.json` 中的
`status: completed` 和对应结果确定已完成项，不能仅凭旧 JSON 存在判断。

| pipeline 参数 | 对照组合 |
|---|---|
| dbnetpp_svtr | DBNet++ + SVTR |
| ours_detector_ours_recognizer | Ours detector + Ours recognizer |
| dbnetpp_ours_recognizer | DBNet++ + Ours recognizer |
| ours_detector_svtr | Ours detector + SVTR |

先完成第 6.0 节共同设置和第 9.1 节冻结哈希检查，再仅补未完成组合。
以下示例只补 Ours detector + Ours recognizer 的 test；把 `pipeline`
改为上表中实际未完成组合的参数名，把 `split` 改为 `validation` 或 `test`。
可以逐项使用同一完整命令恢复全部四条 pipeline，参数与首次运行保持一致：

```bash
pipeline=ours_detector_ours_recognizer
split=test

python3 -u scripts/run_e2e_ocr_experiments.py \
  --plan configs/experiments/e2e_ocr.yaml \
  --pipelines "${pipeline}" \
  --splits "${split}" \
  --python-bin python3 \
  --output-root "${OCR_MANCHU_OUTPUT_ROOT}" \
  --device cuda \
  --checkpoint-policy best_only \
  --fail-fast \
  2>&1 | tee -a "${OCR_MANCHU_OUTPUT_ROOT}/run_logs/11_e2e_four_pipelines.log"
```

如果某条 pipeline 的两个 split 都未完成，可以把 `--splits "${split}"`
替换为 `--splits validation test`；不要把已完成的 test 再作为选参依据。
局部补跑仍执行无退化、IoU 0.75 与严格识别指标。

**补齐后必须重建 8 行 E2E 汇总。** 当前 runner 的 `latest` CSV/JSON
只包含最近一次调用成功完成的组合；局部补跑会将它更新成局部汇总，
但不会删除其他 pipeline 目录内的结果。因此在确认八个组合都是本轮
同一冻结协议后，执行以下命令合并已有结果，不再进行模型推理。
任何结果缺失或与当前 pipeline 的 checkpoint、manifest、固定 IoU 或严格指标
标记不一致都会报错停止；文件内容哈希仍以冻结记录为准。

```bash
python3 - <<'PY'
import os
import runpy
from datetime import datetime, timezone
from pathlib import Path

namespace = runpy.run_path("scripts/run_e2e_ocr_experiments.py")
plan = namespace["load_yaml"]("configs/experiments/e2e_ocr.yaml")
output_root = Path(os.environ["OCR_MANCHU_OUTPUT_ROOT"])
evaluations, issues = namespace["build_pipeline_evaluations"](
    plan, output_root, "best_only"
)
if issues or len(evaluations) != 4:
    raise RuntimeError(f"E2E preflight failed: {issues}")
manifests = namespace["resolve_manifests"](plan)
rows = []
for evaluation in evaluations:
    for split in ("validation", "test"):
        result_path = namespace["result_path_for"](
            output_root, plan, evaluation, split
        )
        result = namespace["validate_result"](
            result_path, evaluation, manifests[split], split
        )
        rows.append(namespace["comparison_row"](result, result_path))
assert len(rows) == 8
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
json_path, csv_path = namespace["write_comparison"](
    rows, output_root / plan["outputs"]["metrics_subdir"], timestamp
)
print(f"[OK] merged_runs={len(rows)} json={json_path} csv={csv_path}")
PY
```

## 12. 完成判据

在写论文表格前逐项确认：

- 16/16 识别模型均有 `best.pth`、`eval_val.json`、`eval_test.json`；
- 13/13 检测模型均有 `best.pth`、`eval_val.json`、`eval_test.json`；
- 所有正式结果 JSON 都包含模型名、split、checkpoint、manifest、指标、运行时间和运行环境；
- SVTRv2/DCM 的正式结果不再出现 `num_batches=0`；
- SVTRv2+NRTR 已出现在 recognition comparison；
- λ 汇总包含 6 行，且 λ=0.1/其他 λ 的指标逻辑符合既定要求；
- VSAA counterpart 汇总包含 6 行；
- 最终 Ours recognition 训练历史可画出 CTC、OTP、total 三条训练曲线；
- E2E 汇总包含 8 行，`matched_text_samples` 大于 0，且所有 pipeline 的 IoU 均为 0.75；
- test 后没有再修改模型、λ、固定 IoU、解码或严格指标规则；
- LMM/API 结果未纳入本轮汇总。

若 24 GB 显存出现 OOM，先降低**评估** batch size（识别 192→128，检测 8→4），这不会改变指标逻辑。正式**训练** batch size 属于受控变量；如必须调整，应对同一对比组所有模型作一致调整并从头重跑该组，不能只改单个模型。
