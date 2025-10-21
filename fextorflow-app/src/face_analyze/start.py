import subprocess
import os
import time
from threading import Thread

def run_fastapi():
    """รัน FastAPI backend"""
    print("Starting FastAPI backend on port 8000...")
    subprocess.run([
        "uvicorn", "main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ])

def run_react():
    """รัน React frontend"""
    print("Starting React frontend on port 5173...")
    # รอ FastAPI start ขึ้นก่อน
    time.sleep(3)
    
    # เปลี่ยนไปยัง frontend directory
    frontend_path = "../frontend"  # แก้ path ให้ตรงกับโครงสร้างของคุณ
    
    if os.path.exists(frontend_path):
        os.chdir(frontend_path)
        subprocess.run(["npm", "run", "dev"])
    else:
        print(f"Frontend path not found: {frontend_path}")

if __name__ == "__main__":
    # สร้าง thread สำหรับแต่ละ service
    fastapi_thread = Thread(target=run_fastapi)
    react_thread = Thread(target=run_react)
    
    # เริ่ม FastAPI ก่อน
    fastapi_thread.start()
    
    # เริ่ม React
    react_thread.start()
    
    # รอทั้ง 2 threads
    fastapi_thread.join()
    react_thread.join()