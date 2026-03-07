@echo off
echo Agarthai — Live Trading (Streamlit)
cd /d "%~dp0"
streamlit run gui/live_app.py --server.port 8502
pause
