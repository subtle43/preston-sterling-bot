@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create .venv. Is Python installed and on PATH?
    pause
    exit /b 1
  )
  echo Installing dependencies...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo pip install failed.
    pause
    exit /b 1
  )
)

if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo.
  echo Created .env
  echo Paste your Discord bot token into DISCORD_TOKEN= then save the file.
  echo Opening .env in Notepad...
  start notepad ".env"
  echo.
  pause
)

echo Starting the Discord bot. Keep this window open.
echo Press Ctrl+C to stop.
".venv\Scripts\python.exe" bot.py
echo.
pause
