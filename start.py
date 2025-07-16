#!/usr/bin/env python3
"""
MovieMate Startup Script
This script starts the backend API server and provides instructions for the frontend.
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'scikit-learn', 'surprise', 
        'groq', 'python-dotenv', 'json5'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"[ERROR] Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_data_files():
    """Check if required data files exist"""
    data_files = [
        "data/ratings_small.csv",
        "data/movies_metadata.csv", 
        "data/credits.csv",
        "data/keywords.csv"
    ]
    
    missing_files = []
    for file_path in data_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"[ERROR] Missing data files: {', '.join(missing_files)}")
        print("Please ensure all data files are present in the data/ directory")
        return False
    
    return True

def start_backend():
    """Start the FastAPI backend server"""
    print("[INFO] Starting MovieMate backend server...")
    
    # Set environment variables
    os.environ['PYTHONPATH'] = os.getcwd()
    
    try:
        # Start the API server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "src.api_server:app", 
            "--host", "0.0.0.0", 
            "--port", "5000", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Backend server stopped by user")
    except Exception as e:
        print(f"[ERROR] Failed to start backend server: {e}")

def start_frontend():
    """Start the React frontend development server"""
    print("[INFO] Starting MovieMate frontend...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("[ERROR] Frontend directory not found")
        return
    
    try:
        os.chdir(frontend_dir)
        subprocess.run(["npm", "run", "dev"], check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Frontend server stopped by user")
    except Exception as e:
        print(f"[ERROR] Failed to start frontend server: {e}")

def main():
    """Main startup function"""
    print("🎬 MovieMate - AI Movie Recommendation System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        sys.exit(1)
    
    print("[INFO] All dependencies and data files found!")
    print("\n[INFO] Starting MovieMate services...")
    print("[INFO] Backend will be available at: http://localhost:5000")
    print("[INFO] Frontend will be available at: http://localhost:5173")
    print("\n[INFO] Press Ctrl+C to stop all services")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait a moment for backend to start
    time.sleep(3)
    
    # Start frontend
    start_frontend()

if __name__ == "__main__":
    main() 