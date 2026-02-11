#!/usr/bin/env python3
"""Serve a LeRobot-backed VLA over HTTP."""

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

from simulator.vla.interface import VLAObservation, create_vla


def create_handler(vla_model, model_name: str):
    
    class LeRobotHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            print(f"[{model_name.upper()}] {args[0]}")
        
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                resp = {"status": "ok", "model": model_name, "type": "lerobot_vla"}
                self.wfile.write(json.dumps(resp).encode())
            else:
                self.send_error(404)
        
        def do_POST(self):
            if self.path == "/predict":
                try:
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data)
                    
                    img_b64 = data.get("image", "")
                    instruction = data.get("instruction", "")
                    proprioception = data.get("proprioception", [0.0] * 8)
                    
                    img_bytes = base64.b64decode(img_b64)
                    image = np.array(Image.open(BytesIO(img_bytes)))
                    
                    obs = VLAObservation(
                        rgb_image=image,
                        depth_image=np.zeros((256, 256), dtype=np.float32),
                        instruction=instruction,
                        proprioception=np.array(proprioception, dtype=np.float32)
                    )
                    
                    action = vla_model.predict(obs)
                    
                    response = {
                        "base_velocity": list(action.base_velocity),
                        "gripper_action": float(action.gripper_action),
                        "done": action.done,
                        "confidence": action.confidence,
                        "control_mode": action.control_mode,
                    }
                    
                    if action.arm_action is not None:
                        response["arm_action"] = action.arm_action.tolist()
                    if action.joint_velocities is not None:
                        response["joint_velocities"] = action.joint_velocities.tolist()
                    if action.joint_positions is not None:
                        response["joint_positions"] = action.joint_positions.tolist()
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                    
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            
            elif self.path == "/predict_batch":
                try:
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data)
                    
                    observations = []
                    for item in data.get("batch", []):
                        img_bytes = base64.b64decode(item.get("image", ""))
                        image = np.array(Image.open(BytesIO(img_bytes)))
                        obs = VLAObservation(
                            rgb_image=image,
                            depth_image=np.zeros((256, 256), dtype=np.float32),
                            instruction=item.get("instruction", ""),
                            proprioception=np.array(item.get("proprioception", [0.0] * 8), dtype=np.float32)
                        )
                        observations.append(obs)
                    
                    if hasattr(vla_model, 'predict_batch'):
                        actions = vla_model.predict_batch(observations)
                    else:
                        actions = [vla_model.predict(obs) for obs in observations]
                    
                    results = []
                    for action in actions:
                        resp = {
                            "base_velocity": list(action.base_velocity),
                            "gripper_action": float(action.gripper_action),
                            "done": action.done,
                            "confidence": action.confidence,
                            "control_mode": action.control_mode,
                        }
                        if action.arm_action is not None:
                            resp["arm_action"] = action.arm_action.tolist()
                        if action.joint_velocities is not None:
                            resp["joint_velocities"] = action.joint_velocities.tolist()
                        results.append(resp)
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"results": results}).encode())
                    
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            else:
                self.send_error(404)
    
    return LeRobotHandler


def main():
    parser = argparse.ArgumentParser(description="LeRobot VLA Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5700)
    parser.add_argument("--model", default=os.getenv("LEROBOT_MODEL", "smolvla"),
                       choices=["smolvla", "pi0", "groot"])
    parser.add_argument("--control-mode", default="low_level",
                       choices=["low_level", "high_level"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"LeRobot VLA Server ({args.model.upper()})")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Control mode: {args.control_mode}")
    print(f"Device: {args.device}")
    print()
    
    print("Loading model...")
    vla = create_vla(
        args.model,
        device=args.device,
        control_mode=args.control_mode
    )
    print(f"Loaded: {vla.model_name}")
    
    print(f"\nStarting server on {args.host}:{args.port}")
    print("Endpoints:")
    print("  POST /predict       - Single prediction")
    print("  POST /predict_batch - Batch prediction")
    print("  GET  /health        - Health check")
    print()
    
    handler = create_handler(vla, args.model)
    server = HTTPServer((args.host, args.port), handler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
