#!/usr/bin/env python3
"""
VLM Planner Server

Runs CogVLM2 as a high-level task planner for robot control.
Analyzes images + instructions to decide NAV vs MANIP mode.

Usage:
    python vlm_planner_server.py --host 0.0.0.0 --port 5600
    
    # Or with specific model:
    python vlm_planner_server.py --model THUDM/cogvlm2-llama3-chat-19B
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import base64
from io import BytesIO
import numpy as np
from PIL import Image

from simulator.vla.vlm_planner import VLMPlanner, VLAObservation


def create_handler(planner: VLMPlanner):
    """Create HTTP request handler with planner reference."""
    
    class VLMPlannerHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            print(f"[VLM] {args[0]}")
        
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "ok", "model": "CogVLM2"}')
            else:
                self.send_error(404)
        
        def do_POST(self):
            if self.path == "/plan":
                try:
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data)
                    
                    img_b64 = data.get("image", "")
                    instruction = data.get("instruction", "")
                    
                    img_bytes = base64.b64decode(img_b64)
                    image = np.array(Image.open(BytesIO(img_bytes)))
                    
                    obs = VLAObservation(
                        rgb_image=image,
                        depth_image=np.zeros((256, 256), dtype=np.float32),
                        instruction=instruction,
                        proprioception=np.zeros(8, dtype=np.float32)
                    )
                    
                    result = planner.plan(obs)
                    
                    response = {
                        "mode": result.mode.value,
                        "subgoal": result.subgoal,
                        "target": result.target_object or result.target_location,
                        "reasoning": result.reasoning,
                        "confidence": result.confidence
                    }
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                    
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            else:
                self.send_error(404)
    
    return VLMPlannerHandler


def main():
    parser = argparse.ArgumentParser(description="VLM Planner Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5600, help="Port to listen on")
    parser.add_argument("--model", default="THUDM/cogvlm2-llama3-chat-19B",
                       help="HuggingFace model ID")
    parser.add_argument("--no-4bit", action="store_true", 
                       help="Disable 4-bit quantization (uses more VRAM)")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()
    
    print("=" * 60)
    print("VLM Planner Server (CogVLM2)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"4-bit quantization: {not args.no_4bit}")
    print()
    
    print("Loading VLM planner...")
    planner = VLMPlanner(
        model_id=args.model,
        device=args.device,
        use_4bit=not args.no_4bit
    )
    
    if not planner.load():
        print("ERROR: Failed to load VLM planner")
        sys.exit(1)
    
    print(f"\nStarting server on {args.host}:{args.port}")
    print("Endpoints:")
    print(f"  POST /plan     - Get task plan from image + instruction")
    print(f"  GET  /health   - Health check")
    print()
    
    handler = create_handler(planner)
    server = HTTPServer((args.host, args.port), handler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
