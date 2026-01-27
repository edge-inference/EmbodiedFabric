#!/usr/bin/env python3
"""
CogACT Inference Server

Keeps CogACT loaded in GPU and serves actions over HTTP.

Usage:
  export HF_TOKEN=hf_xxx
  CUDA_VISIBLE_DEVICES=0 python examples/cogact_server.py --port 5500
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image
import json_numpy
import torch
from vla import load_vla


class CogACTHandler(BaseHTTPRequestHandler):
    model = None

    def _send_json(self, payload, status=200):
        data = json_numpy.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        """Health check endpoint"""
        if self.path == "/health":
            self._send_json({"status": "ok", "model": "CogACT"})
            return
        self._send_json({"error": "unknown endpoint"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json_numpy.loads(body)
        except Exception:
            self._send_json({"error": "invalid json"}, status=400)
            return

        if self.path == "/act":
            image = data.get("image")
            instruction = data.get("instruction", "")
            action = self._predict_one(image, instruction)
            self._send_json({"action": action})
            return

        if self.path == "/act_batch":
            images = data.get("images", [])
            instructions = data.get("instructions", [])
            actions = self._predict_batch(images, instructions)
            self._send_json({"actions": actions})
            return

        self._send_json({"error": "unknown endpoint"}, status=404)

    def _predict_one(self, image, instruction):
        img = Image.fromarray(np.array(image).astype(np.uint8))
        chunk, _ = self.model.predict_action(
            img,
            instruction,
            unnorm_key="fractal20220817_data",
            cfg_scale=1.5,
            use_ddim=True,
            num_ddim_steps=10,
        )
        return np.array(chunk[0]).astype(float).tolist()

    def _predict_batch(self, images, instructions):
        pil_images = [Image.fromarray(np.array(img).astype(np.uint8)) for img in images]
        if hasattr(self.model, "predict_action_batch"):
            chunks, _ = self.model.predict_action_batch(
                pil_images,
                instructions,
                unnorm_key="fractal20220817_data",
                cfg_scale=1.5,
                use_ddim=True,
                num_ddim_steps=10,
            )
        else:
            chunks = []
            for img, inst in zip(pil_images, instructions):
                chunk, _ = self.model.predict_action(
                    img,
                    inst,
                    unnorm_key="fractal20220817_data",
                    cfg_scale=1.5,
                    use_ddim=True,
                    num_ddim_steps=10,
                )
                chunks.append(chunk)
        return [np.array(chunk[0]).astype(float).tolist() for chunk in chunks]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5500)
    parser.add_argument("--model-id", default="CogACT/CogACT-Small")
    parser.add_argument("--action-model-type", default="DiT-S")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print("Loading CogACT model...")
    model = load_vla(
        args.model_id,
        load_for_training=False,
        action_model_type=args.action_model_type,
        future_action_window_size=15,
    )
    if hasattr(model, "vlm"):
        model.vlm = model.vlm.to(torch.bfloat16)
    model.to(args.device).eval()

    CogACTHandler.model = model
    server = ThreadingHTTPServer((args.host, args.port), CogACTHandler)
    print(f"CogACT server listening on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
