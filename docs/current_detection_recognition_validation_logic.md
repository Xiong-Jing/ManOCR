# 当前检测与识别验证/测试逻辑

本文记录当前代码中唯一有效的正式评估协议。旧的模型专属容错距离、字符位置约束、验证退化程度和 Ours detector 的 0.70 IoU 口径已经移除，不应再用于新结果。

## 1. 统一原则

- validation 与 test 使用同一个 evaluator、同一套指标函数和同一套后处理；两者只更换数据清单。
- 模型选择只使用 validation；held-out test 在配置与 checkpoint 确定后运行。
- 每个结果记录模型名、split、checkpoint、manifest、指标、运行时间与运行环境。
- 训练增强不受本次修改影响；取消的是 validation/test 阶段的人为退化。

## 2. 检测

所有 13 个检测配置都遵循：

1. 读取原始 validation/test 图像，不启用 `eval_degradation`。
2. 使用各模型 YAML 中的 DB 后处理参数生成预测框。
3. 预测框和 ground-truth 框做一对一贪心匹配。
4. 固定 `IoU >= 0.75` 为 TP；其余预测为 FP，未匹配 GT 为 FN。
5. 输出 precision、recall、F-measure、TP、FP、FN、GT 数和预测框数。

`scripts/train_detection.py` 的训练期验证和 `scripts/eval_detection.py` 的独立 validation/test 都强制执行上述协议。检测 YAML 不再包含 `eval_degradation`，所有 `eval.iou_thresh` 均为 `0.75`；命令行也不再提供改变正式 IoU 或启用退化评估的入口。

## 3. 识别

对每个 prediction/reference 计算最优 Levenshtein 对齐，并累计：

- `S`：substitution 数；
- `D`：deletion 数；
- `I`：insertion 数；
- `N`：全数据集 reference transcription 的字符总数。

统一指标为：

```text
WA  = exact_matches / samples
CA  = (N - S - D) / N
CER = (S + D + I) / N
```

说明：

- WA 只接受完整字符串完全相等；不允许编辑距离容错，也没有首、次、尾字符例外。
- CA 只衡量 reference 中正确保留的字符，因此 insertion 不降低 CA。
- CER 是标准 corpus-level edit-distance CER，直接从 `S+D+I` 计算，不由 `1-CA` 推导，也不逐样本扣减误差。
- 解码器的 lexicon correction/CTC rerank 属于模型推理过程，不是指标容错；启用时仍以最终 prediction 按上述严格公式评分。

`scripts/train_recognition.py`、`scripts/eval_recognition.py`、根目录 `score_recognition.py` 以及 E2E evaluator 共用这一口径。识别 YAML 不再包含 `metrics` 容错块。

## 4. SVTR + DAB + OTP

主配置为 `configs/recognition/svtr_official_dab_lortho.yaml`。OTP 项从第 1 轮开始使用固定权重：

```text
lambda_OTP = 0.1
L_total = L_CTC + lambda_align * L_align + 0.1 * L_OTP
```

训练代码不再读取 OTP 的 `final`、`decay`、`activate_epoch` 或 `warmup_epochs` 字段。恢复训练和补做最终验证时同样使用固定 `0.1`。lambda sensitivity 配置仍保留不同常数值，目的仅是论文消融；每一个取值也从第 1 轮起恒定，并使用同一严格指标。

## 5. E2E OCR

四条 detector × recognizer pipeline 统一遵循：

- 只使用 detector 预测框裁剪识别输入，不使用 GT boxes；
- 原始整页图像，无评估退化；
- 框匹配固定 `IoU=0.75`；
- E2E WA 要求检测匹配且 transcription 完全正确；
- E2E CER 为包含 matched crops、missed GT deletions 与 false-positive insertions 的标准 `(S+D+I)/N`；
- page transcription 按列内自上而下、列间自左到右重建，并报告 page CER/edit similarity。

## 6. 独立重算识别指标

已有两列 `ground_truth`、`prediction` 的 CSV 时无需重新训练：

```bash
python3 score_recognition.py predictions.csv \
  --model MODEL_NAME \
  --split test \
  --output recognition_scores.csv
```

脚本先移除文本形式的 CTC blank。默认 `--space-policy ignore`，即把空格视为 Romanized Manchu 的格式分隔；只有空格代表真实 token boundary 时才使用 `--space-policy count`。
