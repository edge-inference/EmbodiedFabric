#!/usr/bin/env python3
"""
TDW Persistent Server

Keeps TDW build running: persistent server.

Usage:
    # Terminal 1: Start TDW server (keep running)
    python examples/tdw_server.py --port 1071
    
    # Terminal 2: Run simulation connecting to existing build
    python examples/floorplan_4zone.py --vla smolvla --tdw-address localhost --tdw-port 1071
"""

import argparse
import signal
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="TDW Persistent Server")
    parser.add_argument("--port", type=int, default=1071, help="TDW port")
    parser.add_argument("--display", type=str, default=":99", help="X display for headless")
    args = parser.parse_args()
    
    print("=" * 60)
    print("TDW Persistent Server")
    print("=" * 60)
    print(f"Port: {args.port}")
    print()
    
    # Import TDW
    try:
        from tdw.controller import Controller
    except ImportError:
        print("ERROR: TDW not installed. Run: pip install tdw")
        sys.exit(1)
    
    print("Launching TDW build...")
    print("(This will keep running until you press Ctrl+C)")
    print()
    
    # Launch TDW build
    controller = Controller(launch_build=True, port=args.port)
    
    # Send initial setup
    controller.communicate([
        {"$type": "set_target_framerate", "framerate": -1}
    ])
    
    print(f"TDW build running on port {args.port}")
    print("Connect with: --tdw-address localhost --tdw-port {args.port}")
    print()
    print("Press Ctrl+C to stop...")
    
    # Handle shutdown
    def shutdown(sig, frame):
        print("\nShutting down TDW...")
        controller.communicate({"$type": "terminate"})
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # Keep alive
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
