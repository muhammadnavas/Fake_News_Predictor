@echo off
echo ========================================
echo Streamlit App Restart Script
echo ========================================
echo.
echo This script will:
echo 1. Clear Streamlit cache
echo 2. Restart the application cleanly
echo.
pause

echo.
echo [1/2] Clearing Streamlit cache...
streamlit cache clear

echo.
echo [2/2] Starting Streamlit app...
echo.
echo Press Ctrl+C to stop the server
echo.
streamlit run app.py

pause
