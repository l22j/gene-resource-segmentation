@echo off
cd /d "%~dp0"
chcp 65001 >nul

rem ---- 파이썬 찾기 (run_sam_label.bat 과 같은 순서) ----
set "PY="
if defined SAM_PYTHON if exist "%SAM_PYTHON%" set "PY=%SAM_PYTHON%"
if not defined PY if exist "%~dp0python\python.exe" set "PY=%~dp0python\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY (
  echo [SAM] 파이썬을 찾지 못했습니다. Python 을 설치하거나 SAM_PYTHON 환경변수를 지정하세요.
  pause
  exit /b 1
)

"%PY%" download_weights.py %*
pause
