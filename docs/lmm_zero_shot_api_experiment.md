# 通用 OCR 模型的 zero-shot API 实验

## 1. 实验范围

本实验在现有识别数据集的 `validation` 与 held-out `test` word crops 上评估：

- Qwen2.5-VL-7B-Instruct
- GOT-OCR2.0
- PaddleOCR-VL

三者均只做 API 推理，不训练、不微调，也不读取训练集。API 请求只包含待识别图像和模型允许的固定 OCR 指令，`ground_truth` 始终留在本机，响应返回后才用于评分。

配置入口为 `configs/experiments/lmm_zero_shot.yaml`，执行入口为 `scripts/run_lmm_zero_shot.py`。

## 2. 固定调用协议

| 模型 | API 方式 | 固定调用设置 | 自由文本 prompt |
|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | DashScope OpenAI-compatible Chat Completions | `qwen2.5-vl-7b-instruct`，图像 Data URI，temperature=0 | 支持，使用 YAML 中固定 zero-shot prompt |
| GOT-OCR2.0 | 官方 Gradio Space API | `/run_GOT`，`plain texts OCR` | 官方接口不支持自由 prompt，固定为模型原生 OCR task |
| PaddleOCR-VL | PaddleOCR 官方 Python SDK | `PaddleOCR-VL`，关闭 layout detection，`prompt_label=ocr`，temperature=0 | 官方接口使用模型原生 `ocr` prompt label |

论文中应准确写成：对于支持自由指令的模型固定使用同一个 zero-shot prompt；对于不支持自由指令的专用 OCR API，固定使用其官方 plain-text OCR task。不能声称三个接口都接收了同一段自然语言 prompt。

当前配置显式使用用户选定的原始 `PaddleOCR-VL` 模型标识，不会自动替换成 `PaddleOCR-VL-1.5/1.6`。如果以后决定更换版本，必须在查看 test 结果前修改配置、重新完成 validation，并在论文中报告准确模型标识。

官方接口资料：

- Qwen2.5-VL 官方仓库与 DashScope 示例：<https://github.com/QwenLM-corp/Qwen2.5-VL>
- DashScope OpenAI-compatible Chat API：<https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions>
- GOT-OCR2.0 官方仓库：<https://github.com/Ucas-HaoranWei/GOT-OCR2.0>
- GOT-OCR2.0 官方 Space：<https://huggingface.co/spaces/stepfun-ai/GOT_official_online_demo>
- PaddleOCR 官方 API Python SDK：<https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/inference_deployment/serving/paddleocr_official_api/python.en.md>

## 3. 公平评分口径

三个通用模型与项目内全部识别模型采用同一套严格指标：

```text
WA  = 完全正确的 word crops / 全部 word crops
CA  = (N - S - D) / N
CER = (S + D + I) / N
```

其中 `S/D/I` 来自逐样本最优 Levenshtein alignment，`N` 是 reference transcription 的字符总数。空格仍按项目现有 Romanized Manchu 规则视为版式分隔，不计入字符；输出统一做 Unicode NFC、去空白、小写化和撇号规范化。不会自动抽取 Markdown 中的答案、删除解释文字或清除其他标点，因此格式污染仍会进入错误统计。

每个样本的原始响应、规范化 prediction、`S/D/I/N`、延迟和错误类型都会保存，便于人工检查典型失败模式。

## 4. 远程服务器需要自行准备的内容

### 4.1 Python 依赖

在服务器项目根目录执行：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-api.txt
```

`requirements-api.txt` 只增加：

- `paddleocr>=3.6.0`：其 `PaddleOCRClient` 调用托管 API，不在本地加载 PaddleOCR-VL 权重；
- `gradio_client`：调用 GOT-OCR2.0 Gradio API；
- Qwen 适配器只用 Python 标准 HTTP 库，不要求 OpenAI SDK。

这组实验不使用服务器 RTX 4090；显卡仍用于项目已有的训练和本地模型评估。API 实验需要的是稳定的 HTTPS 出站网络、磁盘空间和充足的 API 配额。

### 4.2 账号、密钥和服务

1. 在阿里云百炼开通 `qwen2.5-vl-7b-instruct`，创建 API Key，保存为 `DASHSCOPE_API_KEY`；同时把控制台显示的业务空间/地域 OpenAI-compatible base URL 保存为 `QWEN2_5_VL_BASE_URL`。YAML 保留旧公共域名作为兼容 fallback，但正式实验优先使用当前官方推荐的业务空间专属域名。
2. 在 Paddle AI Studio 获取 Access Token，保存为 `PADDLEOCR_ACCESS_TOKEN`。
3. GOT 官方公开 Space 运行在共享队列上，只建议用来联通和小样本验证。正式全量实验应复制官方 Space 到自己的私有/专用 Space，保持官方代码和 `ucaslcl/GOT-OCR2_0` 权重不变，并将地址保存为 `GOT_OCR2_GRADIO_SOURCE`。私有 Space 还需 `HF_TOKEN`。
4. 确认将数据上传到服务器，并使 `OCR_MANCHU_REC_ROOT/processed/val.txt`、`test.txt` 及其中的所有图像路径可读。
5. 确认你有权把这些满文图像发送给三方托管服务，并遵守各服务的数据与保留政策。

可复制变量模板：

```bash
cp configs/experiments/lmm_zero_shot.env.example ~/lmm_zero_shot.env
# 在工作区外编辑真实 secret，然后：
source ~/lmm_zero_shot.env
```

不要把真实 key 写进 YAML、shell 脚本、日志或版本库。

## 5. 推荐执行顺序

### 5.1 先检查数据，不调用 API

```bash
python3 scripts/run_lmm_zero_shot.py \
  --prepare-only \
  --max-samples 20 \
  --run-tag smoke20
```

### 5.2 检查依赖与环境变量，不调用 API

```bash
python3 scripts/run_lmm_zero_shot.py \
  --dry-run \
  --max-samples 20 \
  --run-tag smoke20
```

### 5.3 只用 validation 做小规模联通测试

建议先逐模型各调用 20 个相同的、长度分层抽取的样本：

```bash
bash scripts/run_lmm_zero_shot.sh \
  --models qwen2_5_vl_7b \
  --splits validation \
  --max-samples 20 \
  --run-tag smoke20

bash scripts/run_lmm_zero_shot.sh \
  --models got_ocr2_0 \
  --splits validation \
  --max-samples 20 \
  --run-tag smoke20

bash scripts/run_lmm_zero_shot.sh \
  --models paddleocr_vl \
  --splits validation \
  --max-samples 20 \
  --run-tag smoke20
```

相同 `run-tag`、split、seed 和 sample cap 会生成同一份共享抽样清单。小样本只能用于排查接口和输出格式，不能冒充全数据集正式结果。

### 5.4 正式 validation

省略 `--max-samples` 才是完整 split：

```bash
bash scripts/run_lmm_zero_shot.sh \
  --splits validation \
  --run-tag full_v1
```

检查各模型的响应、API 错误数、prompt、模型标识和 validation 结果。此时冻结配置和 prompt。

### 5.5 最后运行 held-out test

```bash
bash scripts/run_lmm_zero_shot.sh \
  --splits test \
  --run-tag full_v1
```

也可一次传入 `--splits validation test`；runner 会先完成全部 validation，再开始任何 test 请求。

所有命令默认断点续跑。网络中断或限流后，用完全相同的命令继续即可；已成功样本不会再次计费。只有确实要清空该 `run-tag` 的生成结果时才使用 `--overwrite`。

## 6. 数据规模、费用与限流

项目现有结果记录显示 validation 为 20,621 个 word crops，test 为 20,623 个。若清单未变化，三个模型全量跑完需要：

```text
(20,621 + 20,623) × 3 = 123,732 次 API 请求
```

正式开始前应在服务控制台确认单次价格、免费额度、并发限制、日配额和最大文件限制。runner 启动时会读取真实清单并打印 `maximum_api_calls`，应以当次打印值为准。

默认并发为 Qwen 4、GOT 1、Paddle 2。若出现 429 或队列错误，应降低 `workers`，必要时在 YAML 的 `request_delay_seconds` 增加间隔。不要为了追求速度绕过服务条款。

## 7. 输出与审计

默认正式输出目录为：

```text
outputs/
  metrics/lmm_zero_shot/lmm_zero_shot_v1/<run-tag>/
    manifests/                         # pilot 时的共享固定样本
    <model>/eval_val.json
    <model>/eval_test.json
    <model>/status_val.json
    <model>/status_test.json
    lmm_zero_shot_summary.csv
    lmm_zero_shot_summary.json
    lmm_zero_shot_summary.md
  predictions/lmm_zero_shot/lmm_zero_shot_v1/<run-tag>/<model>/
    predictions_val.jsonl              # 每个成功请求立即落盘，供断点续跑
    predictions_val.csv
    predictions_test.jsonl
    predictions_test.csv
```

每个正式结果至少包含模型名称、`validation/test`、API 模型标识（以 `api://...` 写入 checkpoint 字段）、配置路径、数据清单路径、全部指标、运行时间、prompt/hash、清单 hash、请求统计和输出文件路径。

若任何样本最终为 `api_error`，脚本只生成 `status_*.json`，不会生成可报告的 `eval_*.json`。修复密钥、限流或服务错误后继续相同命令，直到 `status=completed`。这样网络故障不会被错误地计成空 prediction 和模型删除错误。

正式结果完成后，现有论文表格工具会读取默认的 `full_v1` 汇总：

```bash
python3 tools/generate_paper_tables.py \
  --lmm-summary outputs/metrics/lmm_zero_shot/lmm_zero_shot_v1/full_v1/lmm_zero_shot_summary.json
```

## 8. 论文报告建议

结果表至少报告：

| Model | Setting | Split | N samples | WA | CA | CER | API/model ID |
|---|---|---|---:|---:|---:|---:|---|

正文同时说明：

- 全部为 zero-shot、无微调；
- 输入是现有 recognition word crops，而不是 detector 输出；
- prompt 或模型原生 OCR task 的准确内容；
- 严格 WA/CA/CER 及空格规范化；
- API 调用日期、服务模型标识和 GOT Space 的提交版本；
- 典型失败模式，例如输出原生满文而非 Romanized Manchu、空输出、解释/Markdown 污染、替换或删除占主导；
- 托管 API 可能发生后端版本漂移，公开 GOT Space 也不是有 SLA 的批量评测端点，这是实验限制。

## 9. Windows 本机仅做检查

如果 PowerShell 中没有 `python` 命令，使用已安装的 Python Launcher：

```powershell
py -3.13 -m pip install -r requirements-api.txt
py -3.13 scripts/run_lmm_zero_shot.py --prepare-only --max-samples 20 --run-tag smoke20
```

本机路径配置当前仍指向旧用户目录，因此真正检查图片前还需要设置 `OCR_MANCHU_REC_ROOT`，或按远程服务器流程执行。本项目的正式 API 结果仍应在数据完整、网络稳定的远程环境产生。
