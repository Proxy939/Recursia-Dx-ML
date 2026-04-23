#!/usr/bin/env python3
"""
RecursiaDx - Cross-Platform Startup Script
Starts all services: Brain Tumor API, Main ML API, Backend, Frontend
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def print_banner():
    print("\n" + "=" * 60)
    print("   RecursiaDx Full Stack Startup")
    print("=" * 60 + "\n")

def start_service(name, command, cwd=None, wait=3):
    """Start a service in a new terminal/process."""
    print(f"[{name}] Starting...")
    
    try:
        if sys.platform == "win32":
            # Windows: use start command
            subprocess.Popen(
                f'start "{name}" cmd /k "{command}"',
                shell=True,
                cwd=cwd
            )
        elif sys.platform == "darwin":
            # macOS: use osascript to open new Terminal tab
            script = f'tell application "Terminal" to do script "cd {cwd or os.getcwd()} && {command}"'
            subprocess.Popen(["osascript", "-e", script])
        else:
            # Linux: try gnome-terminal, xterm, or konsole
            terminals = [
                ["gnome-terminal", "--", "bash", "-c", f"cd {cwd or os.getcwd()} && {command}; exec bash"],
                ["xterm", "-e", f"cd {cwd or os.getcwd()} && {command}; bash"],
                ["konsole", "-e", f"cd {cwd or os.getcwd()} && {command}; bash"]
            ]
            
            for term_cmd in terminals:
                try:
                    subprocess.Popen(term_cmd)
                    break
                except FileNotFoundError:
                    continue
            else:
                print(f"  Warning: Could not find terminal emulator. Running in background...")
                subprocess.Popen(command, shell=True, cwd=cwd)
        
        time.sleep(wait)
        print(f"  ✓ {name} started\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to start {name}: {e}\n")
        return False

def main():
    # Get project root
    project_root = Path(__file__).parent.resolve()
    
    print_banner()
    
    # Check if we're in the right directory
    if not (project_root / "ml" / "api" / "app.py").exists():
        print("ERROR: Please run this script from the RecursiaDx root directory")
        sys.exit(1)
    
    # Start services
    services = [
        {
            "name": "Brain Tumor API (Port 5002)",
            "command": "python api/brain_tumor_api.py --port 5002" if sys.platform != "win32" else "python api\\brain_tumor_api.py --port 5002",
            "cwd": str(project_root / "ml")
        },
        {
            "name": "ML API (Port 5000)",
            "command": "python api/app.py" if sys.platform != "win32" else "python api\\app.py",
            "cwd": str(project_root / "ml")
        },
        {
            "name": "Backend (Port 5001)",
            "command": "node server.js",
            "cwd": str(project_root / "backend")
        },
        {
            "name": "Frontend (Port 5173)",
            "command": "npm run dev",
            "cwd": str(project_root / "client")
        }
    ]
    
    for i, service in enumerate(services, 1):
        print(f"[{i}/4] Starting {service['name']}...")
        start_service(service['name'], service['command'], service['cwd'])
    
    print("\n" + "=" * 60)
    print("   All Services Started!")
    print("=" * 60)
    print("\n   Frontend:     http://localhost:5173")
    print("   Backend:      http://localhost:5001")
    print("   ML API:       http://localhost:5000")
    print("   Brain Tumor API: http://localhost:5002")
    print("\n   Press Ctrl+C in each window to stop.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
