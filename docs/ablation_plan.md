# Ablation Plan

## Detection Ablation

| ID | Model | VSAA | Asymmetric Shrink |
|---|---|---:|---:|
| D1 | DBNet++ | No | No |
| D2 | DBNet++ + VSAA | Yes | No |
| D3 | DBNet++ + AS | No | Yes |
| D4 | DBNet++ + VSAA + AS | Yes | Yes |

Configs:

- `configs/detection/dbnetpp_official_baseline.yaml`
- `configs/detection/dbnetpp_vsaa.yaml`
- `configs/detection/dbnetpp_asym_shrink.yaml`
- `configs/detection/dbnetpp_vsaa_asym_shrink.yaml`

## VSAA Counterpart Ablation

DBNet++ 的 ResNet18 backbone、ASF/DBFPN、DBHead、训练清单、100 epochs、
AdamW、学习率、scheduler、batch size、input size 和 seed 全部固定，只替换
ASF+fusion 后的 attention/direction module。

| ID | Variant | Direction module | Validation/test logic |
|---|---|---|---|
| V1 | No attention baseline | Identity | Unified strict |
| V2 | Strip Pooling counterpart | Standard two-branch Strip Pooling | Unified strict |
| V3 | Coordinate Attention counterpart | Coordinate Attention | Unified strict |
| V4 | Horizontal-only strip | Horizontal 1×W strip only | Unified strict |
| V5 | Vertical-only strip | Vertical H×1 strip only | Unified strict |
| V6 | VSAA | VSAA | Unified strict |

“No attention”表示没有额外的方向模块，DBNet++ 自身的 ASF 仍然保留。六组
均在原始 validation/test 图像上以 `IoU=0.75` 统一评估。
各模块按其标准结构实现，并不强行做参数量匹配；评估 JSON 的
`runtime.trainable_parameters` 会记录实际可训练参数量，论文表格应一并报告。
Counterpart 结构参考作者公开的
[Strip Pooling（CVPR 2020）](https://openaccess.thecvf.com/content_CVPR_2020/html/Hou_Strip_Pooling_Rethinking_Spatial_Pooling_for_Scene_Parsing_CVPR_2020_paper.html)
和
[Coordinate Attention（CVPR 2021）](https://github.com/houqb/CoordAttention/blob/main/coordatt.py)。

Plan and runner:

- `configs/experiments/det_vsaa_counterparts.yaml`
- `scripts/run_det_vsaa_counterparts.sh`
- `tools/summarize_vsaa_counterparts.py`

该 runner 默认 `AUTO_RESUME=0`，六组从头训练。只在训练中断且已确认
`last.pth` 完整、配置相同的情况下使用：

```bash
AUTO_RESUME=1 bash scripts/run_det_vsaa_counterparts.sh
```

## Recognition Ablation

| ID | Model | DAB | OTP |
|---|---|---:|---:|
| R1 | SVTR | No | No |
| R2 | SVTR + DAB | Yes | No |
| R3 | SVTR + OTP | No | Yes |
| R4 | SVTR + DAB + OTP | Yes | Yes |

Configs:

- `configs/recognition/svtr_official_baseline.yaml`
- `configs/recognition/svtr_official_dab.yaml`
- `configs/recognition/svtr_official_lortho.yaml`
- `configs/recognition/svtr_official_dab_lortho.yaml`

## OTP Lambda Sensitivity (Paper Section 3.3)

六组实验固定 Our recognizer 的 SVTR+DAB 结构、识别数据 split、200 epochs、
AdamW、学习率、scheduler、256×128 input、batch size 和 seed，只改变
Orthographic Transition Penalty（OTP）的权重：

| ID | lambda | DAB | OTP | OTP schedule | Validation/test logic |
|---|---:|:---:|:---:|---|---|
| L1 | 0 | Yes | No | disabled | Unified strict |
| L2 | 0.025 | Yes | Yes | constant from epoch 1 | Unified strict |
| L3 | 0.05 | Yes | Yes | constant from epoch 1 | Unified strict |
| L4 | 0.1 | Yes | Yes | constant from epoch 1 | Unified strict |
| L5 | 0.2 | Yes | Yes | constant from epoch 1 | Unified strict |
| L6 | 0.5 | Yes | Yes | constant from epoch 1 | Unified strict |

`lambda=0` 在配置层设置 `use_orthographic_loss=false`，不是仅把一个仍会
计算的 loss 乘以零。正值实验只设置固定的 `lambda_ortho`，训练代码不再
支持 OTP activate/warm-up/decay。主模型固定使用 `lambda_ortho=0.1`。

所有 lambda 使用同一严格协议：WA 为完全匹配率，CA 为 `(N-S-D)/N`，CER
为 `(S+D+I)/N`。该协议同时用于训练期每 50 轮 validation 选择 `best.pth`、
独立 validation 和 test。

Plan and runner:

- `configs/experiments/rec_otp_lambda_sensitivity.yaml`
- `scripts/run_rec_otp_lambda_sensitivity.sh`
- `tools/summarize_otp_lambda_sensitivity.py`

```bash
bash scripts/run_rec_otp_lambda_sensitivity.sh
```

只有在 validation/test 数值支持时，才能把 `lambda=0.1` 描述为
“selected as a stable trade-off”。若大 lambda 性能下降，“过强先验惩罚罕见
历史变体”应作为结合错误分析的解释，而不是仅凭总表直接断言。
