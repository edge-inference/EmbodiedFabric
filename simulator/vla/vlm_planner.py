"""
VLM High-Level Planner

Uses CogVLM2 (or similar) for visual reasoning and task decomposition.
Runs at 1-3 Hz to parse instructions and route to NAV/MANIP experts.

Modes can be:
- Local inference (GPU)
- Server mode (HTTP endpoint)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from io import BytesIO
import os
import time
import numpy as np
from PIL import Image

from .interface import VLAObservation


class TaskMode(Enum):
    NAVIGATION = "nav"
    MANIPULATION = "manip"
    IDLE = "idle"


@dataclass
class PlannerOutput:
    """Output from VLM planner."""
    mode: TaskMode
    subgoal: str
    target_object: Optional[str] = None
    target_location: Optional[str] = None
    confidence: float = 1.0
    reasoning: Optional[str] = None


class VLMPlanner:
    """
    High-level VLM planner using CogVLM2 or similar.
    
    Analyzes scene + instruction to decide:
    1. Task mode (NAV vs MANIP)
    2. Current subgoal
    3. Target object/location
    """

    SYSTEM_PROMPT = """You are a robot task planner. Given an image of the scene and a task instruction, analyze and respond with:

1. MODE: Either "NAV" (navigation/movement) or "MANIP" (manipulation/grasping)
2. SUBGOAL: The immediate next action to take
3. TARGET: The object or location to interact with
4. REASONING: Brief explanation

Format your response exactly as:
MODE: [NAV or MANIP]
SUBGOAL: [action description]
TARGET: [object or location name]
REASONING: [brief explanation]

Examples:
- "go to the shelf and pick up the red box" → First MODE: NAV, SUBGOAL: navigate to shelf
- "pick up the box" when near box → MODE: MANIP, SUBGOAL: grasp the box
- "push the cart forward" → MODE: MANIP, SUBGOAL: push cart"""

    def __init__(self,
                 model_id: str = "THUDM/cogvlm2-llama3-chat-19B",
                 server_url: Optional[str] = None,
                 device: str = "cuda:0",
                 use_4bit: bool = True):
        self._model_id = model_id
        self._server_url = server_url or os.getenv("VLM_PLANNER_URL")
        self._device = device
        self._use_4bit = use_4bit
        
        self._model = None
        self._tokenizer = None
        self._loaded = False
        
        self._last_plan_time = 0.0
        self._min_plan_interval = 0.3  # Max ~3 Hz
        self._cached_output: Optional[PlannerOutput] = None

    def load(self) -> bool:
        """Load VLM model for local inference."""
        if self._loaded or self._server_url:
            return True
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading VLM planner: {self._model_id}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id,
                trust_remote_code=True
            )
            
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
            }
            
            if self._use_4bit:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                load_kwargs["device_map"] = self._device
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                **load_kwargs
            )
            
            if not self._use_4bit:
                self._model = self._model.to(self._device).eval()
            
            self._loaded = True
            print(f"VLM planner loaded: {self._model_id}")
            return True
            
        except Exception as e:
            print(f"Failed to load VLM planner: {e}")
            return False

    def _parse_response(self, response: str) -> PlannerOutput:
        """Parse VLM response into structured output."""
        mode = TaskMode.MANIPULATION  # Default
        subgoal = ""
        target = None
        reasoning = None
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith("MODE:"):
                mode_str = line.split(":", 1)[1].strip().upper()
                if "NAV" in mode_str:
                    mode = TaskMode.NAVIGATION
                elif "MANIP" in mode_str:
                    mode = TaskMode.MANIPULATION
            elif line.upper().startswith("SUBGOAL:"):
                subgoal = line.split(":", 1)[1].strip()
            elif line.upper().startswith("TARGET:"):
                target = line.split(":", 1)[1].strip()
            elif line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
        
        return PlannerOutput(
            mode=mode,
            subgoal=subgoal,
            target_object=target if mode == TaskMode.MANIPULATION else None,
            target_location=target if mode == TaskMode.NAVIGATION else None,
            reasoning=reasoning
        )

    def _call_server(self, image: np.ndarray, instruction: str) -> str:
        """Call VLM server for inference."""
        import requests
        import base64
        from io import BytesIO
        
        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        prompt = f"{self.SYSTEM_PROMPT}\n\nInstruction: {instruction}"
        
        response = requests.post(
            self._server_url,
            json={"image": img_b64, "prompt": prompt},
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def _call_local(self, image: np.ndarray, instruction: str) -> str:
        """Run local VLM inference."""
        import torch
        
        pil_image = Image.fromarray(image)
        prompt = f"{self.SYSTEM_PROMPT}\n\nInstruction: {instruction}"
        
        # CogVLM2 specific inference
        inputs = self._model.build_conversation_input_ids(
            self._tokenizer,
            query=prompt,
            images=[pil_image],
        )
        
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(self._device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self._device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self._device),
            'images': [[inputs['images'][0].to(self._device).to(torch.bfloat16)]],
        }
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def plan(self, observation: VLAObservation) -> PlannerOutput:
        """
        Generate high-level plan from observation.
        
        Returns cached result if called too frequently (rate limiting).
        """
        current_time = time.time()
        if current_time - self._last_plan_time < self._min_plan_interval:
            if self._cached_output:
                return self._cached_output
        
        self._last_plan_time = current_time
        
        # Ensure image is uint8
        image = observation.rgb_image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        if self._server_url:
            response = self._call_server(image, observation.instruction)
        else:
            if not self._loaded:
                self.load()
            response = self._call_local(image, observation.instruction)
        
        self._cached_output = self._parse_response(response)
        return self._cached_output


class VLMPlannerServer:
    """HTTP server wrapper for VLM planner (for containerized deployment)."""
    
    def __init__(self, planner: VLMPlanner, host: str = "0.0.0.0", port: int = 5600):
        self._planner = planner
        self._host = host
        self._port = port

    def run(self):
        """Start HTTP server."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        import base64
        
        planner = self._planner
        
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/plan":
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data)
                    
                    img_b64 = data.get("image", "")
                    instruction = data.get("instruction", "")
                    
                    img_bytes = base64.b64decode(img_b64)
                    image = np.array(Image.open(BytesIO(img_bytes)))
                    
                    obs = VLAObservation(
                        rgb_image=image,
                        depth_image=np.zeros((256, 256)),
                        instruction=instruction,
                        proprioception=np.zeros(8)
                    )
                    
                    result = planner.plan(obs)
                    
                    response = {
                        "mode": result.mode.value,
                        "subgoal": result.subgoal,
                        "target": result.target_object or result.target_location,
                        "reasoning": result.reasoning
                    }
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                
                elif self.path == "/health":
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"status": "ok"}')
            
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"status": "ok"}')
        
        server = HTTPServer((self._host, self._port), Handler)
        print(f"VLM Planner server running on {self._host}:{self._port}")
        server.serve_forever()
