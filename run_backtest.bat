@echo off
echo Agarthai — Backtest Analytics
cd /d "%~dp0"
streamlit run gui/backtest_app.py --server.port 8501
pause