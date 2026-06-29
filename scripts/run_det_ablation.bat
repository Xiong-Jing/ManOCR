@echo off
setlocal enabledelayedexpansion

if "%PYTHON%"=="" set PYTHON=C:\Users\ahs\anaconda3\envs\ocr_manchu\python.exe

set CONFIGS=configs\detection\dbnetpp_official_baseline.yaml configs\detection\dbnetpp_vsaa.yaml configs\detection\dbnetpp_asym_shrink.yaml configs\detection\dbnetpp_vsaa_asym_shrink.yaml
set NAMES=dbnetpp_official_baseline dbnetpp_vsaa dbnetpp_asym_shrink dbnetpp_vsaa_asym_shrink

for %%C in (%CONFIGS%) do (
    echo [TRAIN] %%C
    "%PYTHON%" scripts\train_detection.py --config %%C
    if errorlevel 1 exit /b 1
)

for %%E in (%NAMES%) do (
    for %%S in (val test) do (
        echo [EVAL] %%E %%S
        "%PYTHON%" scripts\eval_detection.py ^
            --config configs\detection\%%E.yaml ^
            --checkpoint outputs\checkpoints\detection\%%E\best.pth ^
            --split %%S ^
            --save-predictions
        if errorlevel 1 exit /b 1
    )
)

"%PYTHON%" tools\summarize_detection_formal.py
if errorlevel 1 exit /b 1

"%PYTHON%" tools\plot_detection_formal_curves.py
if errorlevel 1 exit /b 1

echo [OK] Detection ablation finished.
