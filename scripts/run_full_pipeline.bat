@echo off
setlocal

pushd "%~dp0.."

if "%PYTHON%"=="" set PYTHON=C:\Users\ahs\anaconda3\envs\ocr_manchu\python.exe
set PYTHONPATH=%CD%\src;%PYTHONPATH%
set OCR_MANCHU_PROJECT_ROOT=%CD%
set OCR_MANCHU_DET_ROOT=C:\Users\ahs\Desktop\Manchu_Detection_Data
set OCR_MANCHU_REC_ROOT=C:\Users\ahs\Desktop\Manchu_Recognition_Data
set OCR_MANCHU_OUTPUT_ROOT=%CD%\outputs

set DET_CONFIG=configs\detection\dbnetpp_vsaa_asym_shrink.yaml
set DET_CKPT=outputs\checkpoints\detection\dbnetpp_vsaa_asym_shrink\best.pth
set REC_CONFIG=configs\recognition\svtr_official_dab_lortho.yaml
set REC_CKPT=outputs\checkpoints\recognition\svtr_official_dab_lortho\best.pth
set FULL_OCR_WORKSPACE=full_ocr_workspace
set FULL_OCR_INPUT=%FULL_OCR_WORKSPACE%\input
set FULL_OCR_JSON=%FULL_OCR_WORKSPACE%\full_page_predictions.json
set FULL_OCR_TEXT_DIR=%FULL_OCR_WORKSPACE%\page_texts
set FULL_OCR_VIS_DIR=%FULL_OCR_WORKSPACE%

if not exist "%FULL_OCR_WORKSPACE%" mkdir "%FULL_OCR_WORKSPACE%"
if not exist "%FULL_OCR_INPUT%" mkdir "%FULL_OCR_INPUT%"
if exist "%FULL_OCR_JSON%" del /q "%FULL_OCR_JSON%"
if exist "%FULL_OCR_TEXT_DIR%" rmdir /s /q "%FULL_OCR_TEXT_DIR%"
if exist "%FULL_OCR_VIS_DIR%\annotated" rmdir /s /q "%FULL_OCR_VIS_DIR%\annotated"
if exist "%FULL_OCR_VIS_DIR%\sequence" rmdir /s /q "%FULL_OCR_VIS_DIR%\sequence"
if exist "%FULL_OCR_VIS_DIR%\combined" rmdir /s /q "%FULL_OCR_VIS_DIR%\combined"
mkdir "%FULL_OCR_TEXT_DIR%"
mkdir "%FULL_OCR_VIS_DIR%\annotated"
mkdir "%FULL_OCR_VIS_DIR%\sequence"
mkdir "%FULL_OCR_VIS_DIR%\combined"

"%PYTHON%" scripts\infer_full_ocr.py ^
    --det-config %DET_CONFIG% ^
    --det-checkpoint %DET_CKPT% ^
    --rec-config %REC_CONFIG% ^
    --rec-checkpoint %REC_CKPT% ^
    --input "%FULL_OCR_INPUT%" ^
    --output %FULL_OCR_JSON% ^
    --text-output-dir %FULL_OCR_TEXT_DIR% ^
    --device cuda
if errorlevel 1 exit /b 1

"%PYTHON%" tools\visualize_full_ocr_sequence.py ^
    --input-json %FULL_OCR_JSON% ^
    --output-dir %FULL_OCR_VIS_DIR% ^
    --text-dir %FULL_OCR_TEXT_DIR% ^
    --font-path C:/Windows/Fonts/msyh.ttc ^
    --side-by-side
if errorlevel 1 exit /b 1

echo [OK] Full OCR inference finished.
echo [OK] Input: %FULL_OCR_INPUT%
echo [OK] JSON: %FULL_OCR_JSON%
echo [OK] Page texts: %FULL_OCR_TEXT_DIR%
echo [OK] Annotated images: %FULL_OCR_VIS_DIR%\annotated
echo [OK] Sequence images: %FULL_OCR_VIS_DIR%\sequence
echo [OK] Combined images: %FULL_OCR_VIS_DIR%\combined

popd
