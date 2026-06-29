# Experiment Protocol

## Hardware

Formal experiments use RTX 4090 settings:

- Recognition batch size: 128
- Detection batch size: 8
- CUDA required

## Detection

Run:

```bat
scripts\run_det_ablation.bat
```

Experiments:

- DBNet++
- DBNet++ + VSAA
- DBNet++ + AS
- DBNet++ + VSAA + AS

## Recognition

Run:

```bat
scripts\run_rec_ablation.bat
```

Experiments:

- SVTR
- SVTR + DAB
- SVTR + Lortho
- SVTR + DAB + Lortho

## Full OCR

Run:

```bat
scripts\run_full_pipeline.bat
```