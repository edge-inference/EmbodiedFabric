"""High-level VLM planner (NAV vs MANIP)."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from io import BytesIO
import os
import time
import logging
from urllib.parse import urljoin
import numpy as np
from PIL import Image

from .interface import VLAObservation

logger = logging.getLogger(__name__)


class TaskMode(Enum):
    NAVIGATION = "nav"
    MANIPULATION = "manip"
    IDLE = "idle"


@dataclass
class PlannerOutput:
    mode: TaskMode
    subgoal: str
    target_object: Optional[str] = None
    target_location: Optional[str] = None
    confidence: float = 1.0
    reasoning: Optional[str] = None


class VLMPlanner:
    """Planner wrapper for CogVLM2 or a /plan HTTP server."""

    SYSTEM_PROMPT = ( #needs to be refined for a better prompt still
        "Return exactly four lines:\n"
        "MODE: NAV|MANIP\n"
        "SUBGOAL: <next step>\n"
        "TARGET: <object or location>\n"
        "REASONING: <short>\n"
    )

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
        if self._loaded or self._server_url:
            return True
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info("Loading VLM planner: %s", self._model_id)
            
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
            logger.info("VLM planner loaded: %s", self._model_id)
            return True
            
        except Exception:
            logger.exception("Failed to load VLM planner")
            return False

    def _parse_response(self, response: str) -> PlannerOutput:
        mode = TaskMode.MANIPULATION
        subgoal = ""
        target: Optional[str] = None
        reasoning: Optional[str] = None
        
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

    def _plan_url(self) -> str:
        if not self._server_url:
            raise RuntimeError("server_url is not set")
        if self._server_url.rstrip("/").endswith("/plan"):
            return self._server_url
        return urljoin(self._server_url.rstrip("/") + "/", "plan")

    def _call_server(self, image: np.ndarray, instruction: str) -> PlannerOutput:
        import requests
        import base64
        
        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        response = requests.post(
            self._plan_url(),
            json={"image": img_b64, "instruction": instruction},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        mode_str = str(data.get("mode", "")).lower()
        if mode_str.startswith("nav"):
            mode = TaskMode.NAVIGATION
        elif mode_str.startswith("manip"):
            mode = TaskMode.MANIPULATION
        else:
            mode = TaskMode.IDLE

        target = data.get("target", None)
        confidence = float(data.get("confidence", 1.0))
        reasoning = data.get("reasoning", None)

        return PlannerOutput(
            mode=mode,
            subgoal=str(data.get("subgoal", "")),
            target_object=str(target) if target and mode == TaskMode.MANIPULATION else None,
            target_location=str(target) if target and mode == TaskMode.NAVIGATION else None,
            confidence=confidence,
            reasoning=str(reasoning) if reasoning is not None else None,
        )

    def _call_local(self, image: np.ndarray, instruction: str) -> str:
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
            self._cached_output = self._call_server(image, observation.instruction)
        else:
            if not self._loaded:
                self.load()
            response = self._call_local(image, observation.instruction)

            self._cached_output = self._parse_response(response)

        return self._cached_output

#need a http server for container based deploys. 