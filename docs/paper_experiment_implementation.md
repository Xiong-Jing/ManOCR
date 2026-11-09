# 论文实验核心代码补齐说明（2026-11-09）

对应用户提供的 `OCR_Manchu_Pattern_Recognition.pdf`。实现依据为方法公式和实验定义，没有录入论文表格成绩、样本数、训练时长或预设提升幅度，也没有改写真实结果。下列说明取代旧文档中与其冲突的协议。

## 1. 本轮补齐与修正

| 论文位置 | 原代码缺口 | 本轮实现 |
|---|---|---|
| §3.2、公式 (2)–(4) | DBNet++ 使用 ResNet-18；旧 VSAA 含额外通道门控和残差细化 | DBNet++ 配置族改为 ResNet-34；增加 `PaperVSAA`，垂直/水平条带响应相加后产生单通道空间权重，输出 `X * As`；旧类保留用于显式 legacy 配置 |
| §3.2.2、公式 (5) | AS 按 x/y 比例缩放，并使用附加 AS loss | 增加按边方向向内偏移的凸多边形裁剪：`Dx=A(1-r²)/L`，与竖直夹角小于 45° 用 Dx，否则用 Dx/3；直接替换 DB 目标，`gamma_as=0`；不改变推理后处理 |
| §3.3.2、公式 (8)–(10) | 转移矩阵只有平滑计数，没有显式 H/S 规则 | 训练集字符对计数 + add-k + 精确八个规则对乘 ρ + 行归一化 + εT floor + 再归一化；ρ=1 保留语料先验；附带来源与矩阵 SHA256 |
| §3.3.2、公式 (13)–(16) | OTP 置信度平方、阈值截断、整个 batch 归一化且分母至少为 1；主模型还用额外对齐损失/词典纠错 | 按序列计算停止梯度的非空白联合置信度，分母为权重和+ε，再对 batch 取平均；log 前截断到 εT；SVTR 配置族关闭额外对齐 loss 和词典纠错；主实验及 OTP 单模块消融均固定 λ=0.1 |
| §4.1 | 识别按词图随机划分，不能保证页面互斥 | 显式 `image_path,source_page_id` 映射；识别继承检测页面所属 split；新增跨检测/识别/E2E 的页面泄漏审计 |
| §4.1.3、表 3 | 各模型增强配置不一致，且不是论文给出的形变 | 统一训练增强 profile：亮度/对比度、双向随机缩放、8×8 控制网格、按图像尺寸设置水平/竖直位移范围及网格尺度平滑；图像与顶点共享同一前向位移场；验证/测试仍不增强 |
| §4.5.5 | 缺少 ρ 消融入口 | 七档 `0.01/0.025/0.05/0.1/0.2/0.5/1`，固定 λ=0.1，200 epochs，同一 SVTR+DAB；生成独立先验/配置/权重路径，训练、验证、测试、CSV/JSON 汇总 |
| §4.5.5 | best checkpoint 只有 WA 比较 | WA 相同时按更低 CER 选择；将 best CER 保存并恢复；新协议拒绝混用旧协议断点和变更后的转移先验 |
| §4.4.3、§4.7 | 大模型输入为词图，64-token 上限；Paddle 使用无版本别名 | 新增整页配置，读取 `e2e_data` 页面与真实转写；固定 PaddleOCR-VL-1.5、启用页面布局、保留原始整页上传、去掉词图输出长度上限；保留缓存/重试，空输出、显式拒答和不可解析文本计为空预测 |
| §4.7、公式 (20) | 页面得分采用字符相似度 | 两阶段与整页 API 均输出 `mean(100*max(0,1-(Sw+Dw+Iw)/Nw))`；原字符编辑相似度保留为独立辅助字段 |
| §4.7、表 14 | 没有 GT crop 对照 | 同一页、同一 recognizer 对象、同一裁剪 padding、预处理和解码，分别运行检测框与人工 GT 框；预测侧包含漏检删除/误检插入；记录逐词结果和 WA/CER 差值 |
| §4.8 | 缺少人工残余错误分类闭环 | 从 E2E records 导出所有错误事件，人工标记六类主因；验证每个事件已审核、证据未改动；支持将同一合并错误的多个事件归为一个实例；按审核实例数统计占比 |
| §4.6 | 缺少可独立调用的参数量统计和分阶段训练计时 | 构造真实网络计数，排除 buffers/EMA/optimizer；可选模型前向计时与 CUDA 峰值；训练历史记录每轮训练和验证秒数，并输出 `training_runtime.json` |

SVTR 消融及 λ/ρ 扫描默认物理 batch size 统一为 128。需要缩小 batch 时，应统一修改整组配置并从头训练。参数量由实际模块计算，未为贴合论文现有数字调整实现。

## 2. 环境与实验数据前提

从项目根目录运行，安装项目后脚本可以导入 `manchu_ocr`：

```bash
pip install -e .
pip install -r requirements.txt
```

Linux 可沿用现有 `scripts/set_remote_paths.sh`；Windows 本地也可用 `py` 代替 `python`。新实验使用 `configs/paths/remote_server.yaml` 的真实数据路径，空的发布用 manifest 不能作为训练输入。

本轮没有正式数据和检查点可用于训练/正式评估。已有论文表格不参与测试，也不代替实验产物。

## 3. 源页面划分与先验

创建真实来源映射 CSV，包含检测页面、E2E 页面和识别裁剪图的路径。同一原始页面的所有裁剪必须使用同一个 `source_page_id`；映射中的路径与 manifest 使用同一写法。不能根据词图文件名随意猜页面 ID。

```csv
image_path,source_page_id
/data/pages/page001.png,archive_page001
/data/words/page001_word01.png,archive_page001
/data/words/page001_word02.png,archive_page001
```

先准备检测划分，再用 `--page-map` 准备识别数据。检测已有 train/val/test 的页面分配优先于识别随机分配；只对未出现在检测集的独立页面按页面随机划分。

```bash
python tools/prepare_recognition_data.py --config configs/paths/remote_server.yaml --page-map /data/source_pages.csv
python tools/audit_page_splits.py --config configs/paths/remote_server.yaml --page-map /data/source_pages.csv
python tools/build_transition_matrix.py --config configs/paths/remote_server.yaml --rho 0.05
```

审计要求所配置 split 非空，检查检测、识别、E2E 的 train/val/test 互斥及跨任务一致性；E2E 本身没有训练划分，仅审计 val/test。识别准备脚本保留显式 `--allow-word-random-split` 兼容旧探索流程，但该方式不是论文协议。

转移先验不包含 blank，不能跨空格计数，训练字符不在 charset 时立即报错。规则严格为 `ae, ea, oe, ev, wi, wo, wu, wv`；`eo, ve, wa, we`、标点及自转移不额外惩罚。`.npy` 同名 `.json` 记录训练 manifest、charset、先验哈希及 ρ/k/εT。新协议训练/评估会检查这些记录。

## 4. ρ 敏感性实验

```bash
# 只打印七组计划，不读训练样本、不训练。
python scripts/run_rec_otp_rho_sensitivity.py --stage all --dry-run

# 根据真实训练集生成七份固定先验与配置。
python scripts/run_rec_otp_rho_sensitivity.py --stage prepare

# 从头训练七个模型。
python scripts/run_rec_otp_rho_sensitivity.py --stage train

# 仅恢复同一轮已存在且协议一致的 last.pth。
python scripts/run_rec_otp_rho_sensitivity.py --stage train --resume

# 独立验证；默认不会访问 test。
python scripts/run_rec_otp_rho_sensitivity.py --stage evaluate --splits val

# 确定协议、模型和 checkpoint 后，运行 held-out test。
python scripts/run_rec_otp_rho_sensitivity.py --stage evaluate --splits test

# 合并已有真实结果，不再推理。
python scripts/run_rec_otp_rho_sensitivity.py --stage summarize --splits val test
```

支持 `--paths-config` 和 `--output-root`。生成文件位于 `<output-root>/experiments/otp_rho/`；检查点和评估结果使用现有 `checkpoints/recognition/<实验名>` 与 `metrics/recognition/<实验名>`。汇总为 `recognition_otp_rho_sensitivity.{json,csv}`。

已有配置、训练文本、charset、矩阵发生变化会拒绝覆盖；训练阶段若已存在 `last.pth` 必须显式续训或换输出目录。汇总核验结果记录中的配置、权重、manifest 和先验哈希，不混入旧结果。ρ=1 不是 λ=0：前者保留语料先验和 OTP，后者移除 OTP。

原 λ 扫描、VSAA counterparts、检测/识别消融入口仍可使用；其相关 YAML 和配置生成器已同步更新。旧文档中的 96 batch、ResNet-18、旧解码和 AS 辅助损失不再代表新协议。

## 5. 整页 LMM/API 实验

```bash
# 在路径配置中准备 e2e_data.test_list，包含本次指定的同一组页面。
python scripts/run_lmm_zero_shot.py --config configs/experiments/lmm_page_zero_shot.yaml --splits test --prepare-only --run-tag paper_page_v2

# 安装 requirements-api.txt 并设置供应商凭据后，检查请求计划但不发请求。
python scripts/run_lmm_zero_shot.py --config configs/experiments/lmm_page_zero_shot.yaml --splits test --dry-run --run-tag paper_page_v2

# 实际调用；本轮代码修改过程中未执行此命令。
python scripts/run_lmm_zero_shot.py --config configs/experiments/lmm_page_zero_shot.yaml --splits test --run-tag paper_page_v2
```

新配置复用现有供应商环境变量，旧词图配置另行保留。不给输入强制补齐到论文中的页面数量。整页 manifest 指向 `image<TAB>annotation_json`，程序在本地按列排序生成 GT，发送给服务的只有原始图像和固定提示/原生任务。缓存身份含页面内容和 GT，变更页面或协议不能沿用旧成功记录。

接口参数按 [DashScope OpenAI 兼容说明](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions) 和 [PaddleOCR Python 客户端说明](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/inference_deployment/serving/paddleocr_official_api/python.en.md) 的结构传递。Qwen 像素参数放在图像 content block 中；记录显式关闭高分辨率模式，Paddle 使用明确的 1.5 model ID。服务端原生解码行为不由本地代码伪装保证，需保留实际服务版本与响应元数据。

**指标边界**：只返回文本的 API 无法产生可验证的 Detection F。新代码不编造检测框或 Detection F；其 WA 是词序列最小编辑对齐后的正确参考词比例，而两阶段 E2E WA 还要求 IoU 匹配。结果字段明确记录该差别。若需要严格空间匹配比较，必须另行收集模型实际返回的词级框与对应文本，不能从转写字符串反推空间定位成绩。字符 CER 不计排版空白；另存保留排版分隔符的 page CER。页面 PTS 统一采用论文公式 (20)，范围 0–100。

空响应、显式拒答、结构化拒答及不可解析输出作为空预测进入统计；网络失败仍作为未完成请求重试，不把网络故障伪装成模型错误。解析规则固定，原始响应完整留存。

## 6. 预测框 vs 人工 GT 框对照

`--manifest` 必须指向选定同一批页面的真实人工词框与转写标注。工具不会自己声称完成“人工重新标注”。两种条件在同一次运行中使用同一个 recognizer。

```bash
python scripts/eval_crop_oracle.py \
  --manifest /data/e2e/test_manual.txt \
  --det-config configs/detection/dbnetpp_vsaa_asym_shrink.yaml \
  --det-checkpoint outputs/checkpoints/detection/dbnetpp_vsaa_asym_shrink/best.pth \
  --rec-config configs/recognition/svtr_official_dab_lortho.yaml \
  --rec-checkpoint outputs/checkpoints/recognition/svtr_official_dab_lortho/best.pth \
  --split test --device cuda --output-dir outputs/metrics/crop_oracle
```

可加 `--preflight-only` 仅检查数据，不加载权重。输出 `eval_test.json`、`records_test.json`，包含每页每词预测、完整字符错误计数、两组 WA/CER、百分点差值、页面/标注/配置/权重哈希。这里重新运行预测框条件以保证页面和设置完全一致，不抄录论文或旧结果里的预测框分数。

## 7. 人工错误归因

先使用现有 `scripts/eval_full_ocr.py --save-records` 得到 `e2e_records_test.json`：

```bash
python tools/analyze_e2e_errors.py export --records /path/e2e_records_test.json --review outputs/review/e2e_errors.csv
# 人工填写 category、reviewed=yes、reviewer；可补充 notes。
python tools/analyze_e2e_errors.py summarize --records /path/e2e_records_test.json --review outputs/review/e2e_errors.csv
```

category 可选：`head_tail_truncation`、`adjacent_column_merge`、`missed_word`、`false_positive_word_region`、`dot_circle_confusion`、`rare_orthographic_variant`。不自动将字符替换判为点圈混淆或罕见拼写；这些必须看图/语境确认。一个实际合并错误产生多个匹配残余事件时，可人工将 `instance_id` 设为同一值；同组须属于同页、同一主因。

汇总要求所有事件均已审核且证据字段未改；分母为人工归类错误实例数，不是页面数或总词数。不会产生论文中预写的类别计数。

## 8. 参数量、运行时间与测试

```bash
python tools/measure_model_cost.py --config configs/detection/dbnetpp_vsaa_asym_shrink.yaml --output outputs/metrics/cost/detector.json
python tools/measure_model_cost.py --config configs/recognition/svtr_official_dab_lortho.yaml --charset /data/processed/charset.txt --output outputs/metrics/cost/recognizer.json
# 可选：提供真实 checkpoint，或增加独立模型前向基准。
python tools/measure_model_cost.py --config configs/detection/dbnetpp_vsaa_asym_shrink.yaml --benchmark --device cuda --warmup 10 --iterations 50 --output outputs/metrics/cost/detector_forward.json
python -m pytest -q
```

未加 `--benchmark` 时只构造并计数，不下载预训练权重或运行训练。可选 `--checkpoint` 从实际保存的 model 配置构造并严格加载。前向基准使用 FP32、独立 warm-up、CUDA 同步，明确排除读图、预处理和解码后处理，不能充当完整 OCR 延迟。

训练历史新增 `train_seconds`、`validation_seconds`；`training_runtime.json` 汇总已计时轮次并标记历史是否完整。`invocation_seconds_including_setup_and_checkpoint_io` 只对应当前进程调用，训练/验证阶段和不包含独立 held-out test。旧历史缺失的秒数不会反推。GPU 峰值与模型前向峰值分别记录，不当作单个模块的增量训练显存。

本轮验证包括公式/梯度、AS 几何、形变图像与标签一致性、页面划分、rho 固定变量与来源哈希、GT 对照、人工分类完整性、参数共享去重、整页 API 模拟缓存/拒答评分，以及真实 ResNet-34 + PaperVSAA 前向和 DB loss 检查。真实 GPU 训练、正式数据评估和在线 API 调用没有执行。

## 9. 与旧实验的兼容性

新配置标记 `paper_v2_20261109`。旧 VSAA 类和比例缩放 AS 函数保留；只有显式 legacy 配置使用它们。新训练、评估和 pipeline 检查点入口拒绝误用旧协议。识别先验发生变化时拒绝续训或评估。

原有第三方对比模型仍是本项目已有实现，不因本轮补充就等同于官方实现或官方预训练成绩。未添加论文未定义的独立泛化、重复随机种子、错误规则扰动等实验结果。模型有效性仍需真实数据和独立实跑验证。

修改前源码备份在 `tmp/paper_code_before.zip`；未包含数据集或训练输出。完整文件清单见本文后附自动清单。
