@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  FULL CHAT CRAWL - resumes from saved cursors, no cap
echo  Safe to close this window: progress is checkpointed every
echo  500 messages and re-running continues where it stopped.
echo ============================================================
echo.

".venv\Scripts\python.exe" scrape_chat.py
if errorlevel 1 (
  echo.
  echo Crawl exited with an error. Progress is saved - re-run to resume.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo  CRAWL DONE - rebuilding the index
echo ============================================================
echo.

".venv\Scripts\python.exe" build_chat_index.py
if errorlevel 1 (
  echo.
  echo Index build failed. raw.jsonl is intact - re-run build_chat_index.py.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo  ALL DONE. Restart the bot to load the new index.
echo ============================================================
".venv\Scripts\python.exe" scrape_chat.py --status
echo.
pause
