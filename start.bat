@echo off
echo 🎬 MovieMate - AI Movie Recommendation System
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv310\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Please create it first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv310\Scripts\activate.bat

REM Check if required packages are installed
echo [INFO] Checking dependencies...
python -c "import fastapi, uvicorn, pandas, sklearn, surprise" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Required packages not installed. Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if data files exist
if not exist "data\ratings_small.csv" (
    echo [ERROR] Data files not found in data\ directory
    pause
    exit /b 1
)

echo [INFO] All checks passed!
echo.
echo [INFO] Starting MovieMate backend server...
echo [INFO] Backend will be available at: http://localhost:5000
echo [INFO] Frontend will be available at: http://localhost:5173
echo.
echo [INFO] Press Ctrl+C to stop the server
echo.

REM Start the backend server
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 5000 --reload

pause 