"""OpenVLA wrapper."""

import time
import logging
from typing import Optional, List
import numpy as np

from .interface import VLAInterface, VLAObservation, VLAAction, VLAMetrics

logger = logging.getLogger(__name__)


class OpenVLAModel(VLAInterface):
    """OpenVLA model wrapper for profiling."""
    
    MODEL_ID = "openvla/openvla-7b"
    
    def __init__(self, 
                 model_id: str = None,
                 device: str = "cuda",
                 quantization: str = "4bit",
                 latency_budget_ms: float = 100.0):
        self._model_id = model_id or self.MODEL_ID
        self._device = device
        self._quantization = quantization
        self._latency_budget_ms = latency_budget_ms
        
        self._model = None
        self._processor = None
        self._loaded = False
        
        self._last_metrics = VLAMetrics()
        self._total_inferences = 0
    
    def load(self) -> bool:
        """Load the model (call once before inference)"""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            import torch
            
            logger.info(f"Loading OpenVLA model: {self._model_id}")
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {cuda_available}, devices: {torch.cuda.device_count()}")
            
            if cuda_available:
                self._device = "cuda"
            else:
                logger.warning("CUDA not available, using CPU (will be slow!)")
                self._device = "cpu"
            
            load_kwargs = {
                "torch_dtype": torch.bfloat16 if cuda_available else torch.float32,
            }
            
            # Only use device_map="auto" if we have CUDA
            if cuda_available:
                load_kwargs["device_map"] = "auto"
            
            # Only use quantization if requested AND CUDA is available
            if self._quantization == "4bit" and cuda_available:
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                    logger.info("Using 4-bit quantization")
                except ImportError:
                    logger.warning("bitsandbytes not available, using FP16")
            elif self._quantization == "8bit" and cuda_available:
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                    logger.info("Using 8-bit quantization")
                except ImportError:
                    logger.warning("bitsandbytes not available, using FP16")
            else:
                logger.info(f"Using {'bfloat16' if cuda_available else 'float32'} (no quantization)")
            
            self._processor = AutoProcessor.from_pretrained(
                self._model_id, 
                trust_remote_code=True
            )
            
            self._model = AutoModelForVision2Seq.from_pretrained(
                self._model_id,
                trust_remote_code=True,
                **load_kwargs
            )
            
            # Move to device if not using device_map
            if not cuda_available:
                self._model = self._model.to(self._device)
            
            self._loaded = True
            logger.info(f"OpenVLA model loaded successfully on {self._device}")
            return True
            
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            logger.error("Install with: pip install transformers torch")
            return False
        except Exception as e:
            logger.error(f"Failed to load OpenVLA: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, observation: VLAObservation) -> VLAAction:
        """Run VLA inference"""
        if not self._loaded:
            if not self.load():
                return VLAAction(done=True, confidence=0.0)
        
        start = time.perf_counter()
        
        try:
            from PIL import Image
            import torch
            
            if observation.rgb_image.dtype == np.uint8:
                image = Image.fromarray(observation.rgb_image)
            else:
                image = Image.fromarray((observation.rgb_image * 255).astype(np.uint8))
            
            prompt = f"In: What action should the robot take? {observation.instruction}\nOut:"
            
            inputs = self._processor(prompt, image, return_tensors="pt")
            # Move to device and convert float tensors to bfloat16 to match model
            inputs = {
                k: v.to(self._device, dtype=torch.bfloat16) if v.dtype == torch.float32 else v.to(self._device)
                for k, v in inputs.items()
            }
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )
            
            action_text = self._processor.decode(outputs[0], skip_special_tokens=True)
            action = self._parse_action(action_text)
            
        except Exception as e:
            logger.error(f"OpenVLA inference failed: {e}")
            action = VLAAction(done=True, confidence=0.0)
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        self._last_metrics = VLAMetrics(
            latency_ms=latency_ms,
            flops=self._estimate_flops(),
            memory_bytes=self._estimate_memory(),
            tokens_processed=256
        )
        
        self._total_inferences += 1
        
        return action
    
    def predict_batch(self, observations: List[VLAObservation]) -> List[VLAAction]:
        """
        Run batched VLA inference.
        
        Returns a list of actions in the same order as observations.
        """
        if not observations:
            return []
        if not self._loaded:
            if not self.load():
                return [VLAAction(done=True, confidence=0.0) for _ in observations]
        
        start = time.perf_counter()
        actions: List[VLAAction] = []
        
        try:
            from PIL import Image
            import torch
            
            images = []
            prompts = []
            for obs in observations:
                if obs.rgb_image.dtype == np.uint8:
                    images.append(Image.fromarray(obs.rgb_image))
                else:
                    images.append(Image.fromarray((obs.rgb_image * 255).astype(np.uint8)))
                prompts.append(f"In: What action should the robot take? {obs.instruction}\nOut:")
            
            inputs = self._processor(prompts, images, return_tensors="pt", padding=True)
            inputs = {
                k: v.to(self._device, dtype=torch.bfloat16) if v.dtype == torch.float32 else v.to(self._device)
                for k, v in inputs.items()
            }
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )
            
            # Decode each output
            for i in range(outputs.shape[0]):
                action_text = self._processor.decode(outputs[i], skip_special_tokens=True)
                actions.append(self._parse_action(action_text))
            
        except Exception as e:
            logger.error(f"OpenVLA batch inference failed: {e}")
            actions = [VLAAction(done=True, confidence=0.0) for _ in observations]
        
        latency_ms = (time.perf_counter() - start) * 1000
        self._last_metrics = VLAMetrics(
            latency_ms=latency_ms,
            flops=self._estimate_flops() * max(1, len(observations)),
            memory_bytes=self._estimate_memory(),
            tokens_processed=256 * max(1, len(observations))
        )
        
        self._total_inferences += len(observations)
        return actions
    
    def _parse_action(self, action_text: str) -> VLAAction:
        """Parse model output into action"""
        action_text = action_text.lower()
        
        if "grasp" in action_text or "pick" in action_text:
            return VLAAction(
                base_velocity=(0.0, 0.0),
                gripper_action=1.0,
                confidence=0.9
            )
        elif "release" in action_text or "place" in action_text:
            return VLAAction(
                base_velocity=(0.0, 0.0),
                gripper_action=0.0,
                confidence=0.9
            )
        elif "forward" in action_text or "move" in action_text:
            return VLAAction(
                base_velocity=(0.5, 0.0),
                gripper_action=0.5,
                confidence=0.8
            )
        elif "left" in action_text:
            return VLAAction(
                base_velocity=(0.0, 0.5),
                gripper_action=0.5,
                confidence=0.8
            )
        elif "right" in action_text:
            return VLAAction(
                base_velocity=(0.0, -0.5),
                gripper_action=0.5,
                confidence=0.8
            )
        elif "stop" in action_text or "done" in action_text:
            return VLAAction(
                base_velocity=(0.0, 0.0),
                done=True,
                confidence=0.9
            )
        else:
            return VLAAction(
                base_velocity=(0.1, 0.0),
                confidence=0.5
            )
    
    def _estimate_flops(self) -> float:
        """Estimate FLOPS for inference"""
        params = 7e9
        tokens = 256
        flops_per_token = 2 * params
        return flops_per_token * tokens
    
    def _estimate_memory(self) -> int:
        """Estimate memory usage"""
        params = 7e9
        if self._quantization == "4bit":
            return int(params * 0.5)
        elif self._quantization == "8bit":
            return int(params * 1)
        else:
            return int(params * 2)
    
    def get_metrics(self) -> VLAMetrics:
        return self._last_metrics
    
    def reset(self) -> None:
        pass
    
    @property
    def model_name(self) -> str:
        return f"OpenVLA-7B ({self._quantization})"
    
    @property
    def latency_budget_ms(self) -> float:
        return self._latency_budget_ms
