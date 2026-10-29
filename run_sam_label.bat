@echo off
cd /d "%~dp0"

rem ---- 파이썬 찾기: SAM_PYTHON 환경변수 > 동봉 python > Anaconda/Miniconda > PATH ----
set "PY="
if defined SAM_PYTHON if exist "%SAM_PYTHON%" set "PY=%SAM_PYTHON%"
if not defined PY if exist "%~dp0python\python.exe" set "PY=%~dp0python\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "%ProgramData%\anaconda3\python.exe" set "PY=%ProgramData%\anaconda3\python.exe"
if not defined PY if exist "%ProgramData%\miniconda3\python.exe" set "PY=%ProgramData%\miniconda3\python.exe"
if not defined PY for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY (
  echo [SAM] 파이썬을 찾지 못했습니다. Anaconda 를 설치하거나 SAM_PYTHON 환경변수에 python.exe 경로를 지정하세요.
  pause
  exit /b 1
)

rem ---- 필요한 패키지 확인 ----
"%PY%" -c "import torch, segment_anything, PIL, numpy" 2>nul
if errorlevel 1 (
  echo [SAM] %PY% 에 필요한 패키지가 없습니다. 아래 명령으로 설치하세요:
  echo     "%PY%" -m pip install -r "%~dp0requirements.txt"
  echo     ^(GPU 사용 시 torch 는 https://pytorch.org 안내대로 CUDA 버전에 맞춰 설치^)
  pause
  exit /b 1
)

echo [SAM] python: %PY%
"%PY%" sam_label_server.py --open %*
pause
