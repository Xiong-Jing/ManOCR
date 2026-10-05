@echo off
setlocal enabledelayedexpansion

if "%PYTHON%"=="" set PYTHON=C:\Users\ahs\anaconda3\envs\ocr_manchu\python.exe

set CONFIGS=configs\recognition\svtr_official_baseline.yaml configs\recognition\svtr_official_dab.yaml configs\recognition\svtr_official_lortho.yaml configs\recognition\svtr_official_dab_lortho.yaml
set NAMES=svtr_official_baseline svtr_official_dab svtr_official_lortho svtr_official_dab_lortho

for %%C in (%CONFIGS%) do (
    echo [TRAIN] %%C
    "%PYTHON%" scripts\train_recognition.py --config %%C
    if errorlevel 1 exit /b 1
)

for %%E in (%NAMES%) do (
    for %%S in (val test) do (
        echo [EVAL] %%E %%S
        "%PYTHON%" scripts\eval_recognition.py ^
            --config configs\recognition\%%E.yaml ^
            --checkpoint outputs\checkpoints\recognition\%%E\best.pth ^
            --split %%S ^
            --save-predictions
        if errorlevel 1 exit /b 1
    )
)

"%PYTHON%" tools\summarize_recognition_formal.py
if errorlevel 1 exit /b 1

"%PYTHON%" tools\plot_recognition_formal_curves.py
if errorlevel 1 exit /b 1

echo [OK] Recognition ablation finished.
