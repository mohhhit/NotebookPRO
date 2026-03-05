@echo off
REM NotebookPRO Launcher for Windows
REM RAG-powered AI study assistant - no training required!

echo ========================================
echo   NotebookPRO - RAG AI Study Assistant
echo   No Training Required!
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo Python found: 
python --version
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Check if requirements are installed
if not exist "venv\Lib\site-packages\streamlit\" (
    echo Installing required packages...
    echo This may take a few minutes...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
    echo.
    echo All packages installed successfully!
    echo.
)

REM Create necessary directories
if not exist "data\" mkdir data
if not exist "data\uploads\" mkdir data\uploads
if not exist "data\vector_db\" mkdir data\vector_db
if not exist "data\chats\" mkdir data\chats
if not exist "models\" mkdir models

REM Check if .env exists, if not copy from .env.example
if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env file from template...
        copy .env.example .env
    )
)

REM Launch the application
echo.
echo ========================================
echo   Starting NotebookPRO...
echo ========================================
echo.
echo The application will open in your browser at:
echo http://localhost:8501
echo.
echo To stop the application, press Ctrl+C
echo.

streamlit run App.py

REM Deactivate virtual environment on exit
deactivate
