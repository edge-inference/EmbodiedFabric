#!/usr/bin/env python3
"""Run a persistent TDW build on a port."""

import argparse
import signal
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="TDW Persistent Server")
    parser.add_argument("--port", type=int, default=1071, help="TDW port")
    args = parser.parse_args()
    
    print("=" * 60)
    print("TDW Persistent Server")
    print("=" * 60)
    print(f"Port: {args.port}")
    print()
    
    try:
        from tdw.controller import Controller
    except ImportError:
        print("ERROR: TDW not installed. Run: pip install tdw")
        sys.exit(1)
    
    print("Launching TDW build...")
    print("(This will keep running until you press Ctrl+C)")
    print()
    
    controller = Controller(launch_build=True, port=args.port)
    
    controller.communicate([
        {"$type": "set_target_framerate", "framerate": -1}
    ])
    
    print(f"TDW build running on port {args.port}")
    print(f"Connect with: --tdw-address localhost --tdw-port {args.port}")
    print()
    print("Press Ctrl+C to stop...")
    
    def shutdown(sig, frame):
        print("\nShutting down TDW...")
        controller.communicate([{"$type": "terminate"}])
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
