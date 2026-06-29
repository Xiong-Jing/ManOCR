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

## Recognition Ablation

| ID | Model | DAB | Lortho |
|---|---|---:|---:|
| R1 | SVTR | No | No |
| R2 | SVTR + DAB | Yes | No |
| R3 | SVTR + Lortho | No | Yes |
| R4 | SVTR + DAB + Lortho | Yes | Yes |

Configs:

- `configs/recognition/svtr_official_baseline.yaml`
- `configs/recognition/svtr_official_dab.yaml`
- `configs/recognition/svtr_official_lortho.yaml`
- `configs/recognition/svtr_official_dab_lortho.yaml`
