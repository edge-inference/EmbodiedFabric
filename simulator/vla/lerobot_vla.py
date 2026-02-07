"""LeRobot VLA Wrappers: SmolVLA, Pi0, GR00T"""

from abc import ABC
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import time
import numpy as np
from PIL import Image

from .interface import VLAInterface, VLAObservation, VLAAction, VLAMetrics


class VLAControlMode:
    HIGH_LEVEL = "high_level"
    LOW_LEVEL = "low_level"


@dataclass
class LeRobotConfig:
    """Configuration for LeRobot VLA models."""
    model_type: str = "smolvla"  # smolvla, pi0, groot
    model_path: str = ""  # HuggingFace path or local
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    compile_model: bool = False
    action_chunk_size: int = 10
    latency_budget_ms: float = 100.0
    control_mode: str = VLAControlMode.LOW_LEVEL


class BaseLeRobotVLA(VLAInterface, ABC):

    def __init__(self, config: LeRobotConfig):
        self._config = config
        self._model = None
        self._policy = None
        self._loaded = False
        self._last_metrics = VLAMetrics()
        self._action_buffer: List[np.ndarray] = []
        self._buffer_idx = 0

    def _load_model(self):
        raise NotImplementedError

    def _preprocess_observation(self, obs: VLAObservation) -> Dict[str, Any]:
        image = obs.rgb_image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_img = Image.fromarray(image)
        
        return {
            "observation.images.camera1": pil_img,
            "observation.images.camera2": pil_img,
            "observation.images.camera3": pil_img,
            "observation.state": obs.proprioception,
            "task": obs.instruction,
        }

    def _postprocess_action(self, raw_action: np.ndarray) -> VLAAction:
        control_mode = self._config.control_mode
        
        if control_mode == VLAControlMode.LOW_LEVEL:
            if len(raw_action) >= 2:
                # GR00T outputs arm joint values, not base velocities.
                # Use magnitude of first dim as forward speed (always forward),
                # second dim as turn signal.
                forward = abs(float(raw_action[0])) * 0.3
                turn = float(raw_action[1]) * 0.5
                base_vel = (forward, turn)
            else:
                base_vel = (0.0, 0.0)
            return VLAAction(
                base_velocity=base_vel,
                arm_action=raw_action,
                gripper_action=float(raw_action[-1]) if len(raw_action) > 0 else 0.5,
                confidence=1.0,
                control_mode=control_mode,
                joint_velocities=raw_action,
                reasoning=f"[{self._config.model_type.upper()}] low-level"
            )
        else:
            if len(raw_action) >= 7:
                base_vel = (float(raw_action[0]), float(raw_action[1]))
                arm_action = raw_action[2:6]
                gripper = float(raw_action[6])
            elif len(raw_action) >= 2:
                base_vel = (float(raw_action[0]), float(raw_action[1]))
                arm_action = None
                gripper = 0.5
            else:
                base_vel = (0.0, 0.0)
                arm_action = raw_action if len(raw_action) > 0 else None
                gripper = 0.5

            return VLAAction(
                base_velocity=base_vel,
                arm_action=arm_action,
                gripper_action=gripper,
                confidence=1.0,
                control_mode=control_mode,
                reasoning=f"[{self._config.model_type.upper()}] high-level"
            )

    @property
    def control_mode(self) -> str:
        return self._config.control_mode
    
    def set_control_mode(self, mode: str) -> None:
        if mode in (VLAControlMode.HIGH_LEVEL, VLAControlMode.LOW_LEVEL):
            self._config.control_mode = mode

    def predict(self, observation: VLAObservation) -> VLAAction:
        if not self._loaded:
            self._load_model()

        start = time.perf_counter()

        if self._action_buffer and self._buffer_idx < len(self._action_buffer):
            action = self._action_buffer[self._buffer_idx]
            self._buffer_idx += 1
        else:
            inputs = self._preprocess_observation(observation)
            raw_actions = self._run_inference(inputs)
            
            if raw_actions.ndim == 2:
                self._action_buffer = [raw_actions[i] for i in range(raw_actions.shape[0])]
            else:
                self._action_buffer = [raw_actions]
            self._buffer_idx = 1
            action = self._action_buffer[0]

        result = self._postprocess_action(action)
        
        self._last_metrics = VLAMetrics(
            latency_ms=(time.perf_counter() - start) * 1000
        )
        
        return result

    def predict_batch(self, observations: List[VLAObservation]) -> List[VLAAction]:
        if not self._loaded:
            self._load_model()
        
        if not observations:
            return []
        
        start = time.perf_counter()
        
        batch = self._prepare_batch_multi(observations)
        raw_actions = self._run_inference_batch(batch)
        
        results = []
        for i in range(len(observations)):
            action = raw_actions[i] if raw_actions.ndim == 2 else raw_actions
            results.append(self._postprocess_action(action))
        
        self._last_metrics = VLAMetrics(
            latency_ms=(time.perf_counter() - start) * 1000
        )
        
        return results

    def _prepare_batch_multi(self, observations: List[VLAObservation]) -> Dict[str, Any]:
        import torch
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        cam1_images, cam2_images, cam3_images = [], [], []
        states = []
        tasks = []
        
        for obs in observations:
            img = obs.rgb_image
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            img_tensor = transform(pil_img)
            cam1_images.append(img_tensor)
            cam2_images.append(img_tensor)
            cam3_images.append(img_tensor)
            states.append(torch.tensor(obs.proprioception, dtype=torch.float32))
            tasks.append(obs.instruction)
        
        return {
            "observation.images.camera1": torch.stack(cam1_images).to(self._config.device),
            "observation.images.camera2": torch.stack(cam2_images).to(self._config.device),
            "observation.images.camera3": torch.stack(cam3_images).to(self._config.device),
            "observation.state": torch.stack(states).to(self._config.device),
            "task": tasks,
        }

    def _run_inference_batch(self, batch: Dict[str, Any]) -> np.ndarray:
        import torch
        with torch.no_grad():
            actions = self._policy.select_action(batch)
            return actions.cpu().numpy()

    def _run_inference(self, inputs: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def get_metrics(self) -> VLAMetrics:
        return self._last_metrics

    def reset(self) -> None:
        self._action_buffer.clear()
        self._buffer_idx = 0

    @property
    def latency_budget_ms(self) -> float:
        return self._config.latency_budget_ms


class SmolVLAModel(BaseLeRobotVLA):
    """SmolVLA: 450M Flow Matching (lerobot/smolvla_base)"""

    DEFAULT_MODEL = "lerobot/smolvla_base"

    def __init__(self,
                 model_path: str = DEFAULT_MODEL,
                 device: str = "cuda:0",
                 latency_budget_ms: float = 50.0,
                 control_mode: str = VLAControlMode.LOW_LEVEL,
                 **kwargs):
        config = LeRobotConfig(
            model_type="smolvla",
            model_path=model_path,
            device=device,
            latency_budget_ms=latency_budget_ms,
            control_mode=control_mode
        )
        super().__init__(config)
        self._preprocess = None
        self._postprocess = None

    def _load_model(self):
        try:
            import torch
            try:
                from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
                from lerobot.policies.factory import make_pre_post_processors
            except ImportError:
                from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
                from lerobot.common.policies.factory import make_pre_post_processors
            
            print(f"Loading SmolVLA from {self._config.model_path}...")
            
            self._policy = SmolVLAPolicy.from_pretrained(self._config.model_path)
            self._policy = self._policy.to(self._config.device)
            self._policy.eval()
            
            # Create pre/post processors (handles tokenization, normalization)
            self._preprocess, self._postprocess = make_pre_post_processors(
                self._policy.config,
                self._config.model_path,
                preprocessor_overrides={"device_processor": {"device": self._config.device}},
            )
            
            if self._config.compile_model:
                self._policy = torch.compile(self._policy)
            
            self._loaded = True
            print("SmolVLA loaded successfully")
            
        except ImportError as e:
            print(f"LeRobot not installed. Install with: pip install lerobot[smolvla]")
            raise e

    def _run_inference(self, inputs: Dict[str, Any]) -> np.ndarray:
        import torch
        
        with torch.no_grad():
            frame = self._build_frame(inputs)
            batch = self._preprocess(frame)
            action = self._policy.select_action(batch)
            action = self._postprocess(action)
            return action.cpu().numpy()

    def _build_frame(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Build inference frame in LeRobot expected format."""
        import torch
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        frame = {}
        
        # State
        state = inputs["observation.state"]
        frame["observation.state"] = torch.tensor(state, dtype=torch.float32)
        
        # Images
        for key in ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"]:
            if key in inputs:
                frame[key] = transform(inputs[key])
        
        # Task instruction
        frame["task"] = inputs.get("task", "")
        
        return frame

    @property
    def model_name(self) -> str:
        return f"SmolVLA({self._config.model_path})"


class Pi0Model(BaseLeRobotVLA):
    """Pi0: 3B VLM + Flow Matching (lerobot/pi0_base), 50Hz motor commands, cross-embodiment trained"""

    DEFAULT_MODEL = "lerobot/pi0_base"

    def __init__(self,
                 model_path: str = DEFAULT_MODEL,
                 device: str = "cuda:0",
                 latency_budget_ms: float = 100.0,
                 control_mode: str = VLAControlMode.LOW_LEVEL,
                 **kwargs):
        config = LeRobotConfig(
            model_type="pi0",
            model_path=model_path,
            device=device,
            latency_budget_ms=latency_budget_ms,
            control_mode=control_mode
        )
        super().__init__(config)

    def _load_model(self):
        try:
            import torch
            try:
                from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            except ImportError:
                from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
            
            print(f"Loading Pi0 from {self._config.model_path}...")
            
            self._policy = PI0Policy.from_pretrained(self._config.model_path)
            self._policy = self._policy.to(self._config.device)
            self._policy.eval()
            
            if self._config.compile_model:
                self._policy = torch.compile(self._policy)
            
            self._loaded = True
            print("Pi0 loaded successfully")
            
        except ImportError as e:
            print(f"LeRobot Pi0 not installed. Install with: pip install lerobot[pi]")
            raise e

    def _run_inference(self, inputs: Dict[str, Any]) -> np.ndarray:
        import torch
        
        with torch.no_grad():
            batch = self._prepare_batch(inputs)
            action = self._policy.select_action(batch)
            return action.cpu().numpy()

    def _prepare_batch(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        
        state = inputs["observation.state"]
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._config.device)
        
        batch = {"observation.state": state_tensor}
        
        for key in ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"]:
            if key in inputs:
                img_tensor = transform(inputs[key]).unsqueeze(0).to(self._config.device)
                batch[key] = img_tensor
        
        # Tokenize instruction for Pi0
        task_text = inputs.get("task", "")
        if hasattr(self._policy, 'processor') and self._policy.processor is not None:
            tokens = self._policy.processor.tokenizer(
                task_text,
                return_tensors="pt",
                padding="max_length",
                max_length=48,
                truncation=True
            )
            batch["observation.language.tokens"] = tokens["input_ids"].to(self._config.device)
            if "attention_mask" in tokens:
                batch["observation.language.attention_mask"] = tokens["attention_mask"].to(self._config.device)
        
        return batch

    @property
    def model_name(self) -> str:
        return f"Pi0({self._config.model_path})"


class GR00TModel(BaseLeRobotVLA):
    """GR00T N1.5: Eagle VLM + DiT (nvidia/GR00T-N1.5-3B)"""

    DEFAULT_MODEL = "nvidia/GR00T-N1.5-3B"

    def __init__(self,
                 model_path: str = DEFAULT_MODEL,
                 device: str = "cuda:0",
                 latency_budget_ms: float = 150.0,
                 control_mode: str = VLAControlMode.LOW_LEVEL,
                 **kwargs):
        config = LeRobotConfig(
            model_type="groot",
            model_path=model_path,
            device=device,
            latency_budget_ms=latency_budget_ms,
            control_mode=control_mode
        )
        super().__init__(config)
        self._preprocess = None
        self._postprocess = None

    def _load_model(self):
        try:
            import torch
            try:
                from lerobot.policies.groot.modeling_groot import GrootPolicy
                from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
            except ImportError:
                from lerobot.common.policies.groot.modeling_groot import GrootPolicy
                from lerobot.common.policies.groot.processor_groot import make_groot_pre_post_processors
            
            print(f"Loading GR00T from {self._config.model_path}...")
            
            self._policy = GrootPolicy.from_pretrained(self._config.model_path)
            self._policy = self._policy.to(self._config.device)
            self._policy.eval()
            
            # Build processor pipeline directly (HF hub lacks policy_preprocessor.json)
            try:
                self._policy.config.device = self._config.device
                self._preprocess, self._postprocess = make_groot_pre_post_processors(
                    self._policy.config
                )
                print(f"GR00T processor built: embodiment={self._policy.config.embodiment_tag}")
            except Exception as e:
                print(f"Warning: Could not build processor: {e}")
                self._preprocess = None
                self._postprocess = None
            
            self._loaded = True
            print("GR00T loaded successfully")
            
        except ImportError as e:
            print(f"LeRobot GR00T not installed. Install with: pip install lerobot[groot]")
            raise e

    def _run_inference(self, inputs: Dict[str, Any]) -> np.ndarray:
        import torch
        
        with torch.no_grad():
            if self._preprocess is not None:
                frame = self._build_frame(inputs)
                batch = self._preprocess(frame)
                action = self._policy.select_action(batch)
                if self._postprocess is not None:
                    action = self._postprocess(action)
            else:
                batch = self._prepare_batch(inputs)
                action = self._policy.select_action(batch)
            return action.cpu().numpy()

    def _build_frame(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Build inference frame in LeRobot expected format."""
        import torch
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        frame = {}
        frame["observation.state"] = torch.tensor(inputs["observation.state"], dtype=torch.float32)
        
        for key in ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"]:
            if key in inputs:
                frame[key] = transform(inputs[key])
        
        frame["task"] = inputs.get("task", "")
        return frame

    def _prepare_batch(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare batch with Eagle processor for GR00T model."""
        import torch
        from transformers import AutoProcessor
        from PIL import Image
        import numpy as np_module
        
        device = self._config.device
        
        if not hasattr(self, '_eagle_processor'):
            processor_path = '/home/modfi/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5'
            self._eagle_processor = AutoProcessor.from_pretrained(
                processor_path, trust_remote_code=True
            )
        
        images = []
        for key in ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"]:
            if key in inputs:
                img = inputs[key]
                if isinstance(img, Image.Image):
                    images.append(img)
                elif isinstance(img, np_module.ndarray):
                    images.append(Image.fromarray(img.astype(np_module.uint8)))
        
        if not images:
            img = inputs.get("observation.images.camera1")
            if img is None:
                img = np_module.zeros((224, 224, 3), dtype=np_module.uint8)
            images = [Image.fromarray(img.astype(np_module.uint8)) if isinstance(img, np_module.ndarray) else img]
        
        task = inputs.get("task", "pick up the object")
        
        content = [{'type': 'image', 'image': img} for img in images]
        content.append({'type': 'text', 'text': task})
        conversation = [{'role': 'user', 'content': content}]
        
        text = self._eagle_processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        img_inputs, _ = self._eagle_processor.process_vision_info(conversation)
        
        eagle_out = self._eagle_processor(
            text=text,
            images=img_inputs,
            return_tensors="pt",
            padding=True,
        )
        
        batch = {}
        for k, v in eagle_out.items():
            batch[f"eagle_{k}"] = v.to(device) if hasattr(v, 'to') else v
        
        state = inputs["observation.state"]
        state_array = np_module.array(state, dtype=np_module.float32)
        
        max_state_dim = 64
        if len(state_array) < max_state_dim:
            padded_state = np_module.zeros(max_state_dim, dtype=np_module.float32)
            padded_state[:len(state_array)] = state_array
            state_array = padded_state
        
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        batch["state"] = state_tensor
        
        state_mask = torch.ones(state_tensor.shape[:2], dtype=torch.bool, device=device)
        batch["state_mask"] = state_mask
        
        batch["embodiment_id"] = torch.tensor([31], dtype=torch.long, device=device)
        
        return batch

    def predict_batch(self, observations: List[VLAObservation]) -> List[VLAAction]:
        """GR00T requires Eagle processor per observation (no native batch support)."""
        return [self.predict(obs) for obs in observations]

    @property
    def model_name(self) -> str:
        return f"GR00T({self._config.model_path})"


class LeRobotServerClient(VLAInterface):
    """Client for LeRobot VLA server (SmolVLA/Pi0/GR00T via HTTP)."""

    def __init__(self,
                 server_url: str = None,
                 model_type: str = "smolvla",
                 control_mode: str = VLAControlMode.LOW_LEVEL,
                 latency_budget_ms: float = 100.0,
                 **kwargs):
        import os
        self._server_url = server_url or os.getenv("LEROBOT_SERVER_URL", "http://localhost:5700/predict")
        self._batch_url = self._server_url.replace("/predict", "/predict_batch")
        self._model_type = model_type
        self._control_mode = control_mode
        self._latency_budget_ms = latency_budget_ms
        self._last_metrics = VLAMetrics()

    def predict(self, observation: VLAObservation) -> VLAAction:
        import requests
        import base64
        from io import BytesIO
        
        start = time.perf_counter()
        
        image = observation.rgb_image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_img = Image.fromarray(image)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        response = requests.post(
            self._server_url,
            json={
                "image": img_b64,
                "instruction": observation.instruction,
                "proprioception": observation.proprioception.tolist()
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        self._last_metrics = VLAMetrics(latency_ms=(time.perf_counter() - start) * 1000)
        
        return VLAAction(
            base_velocity=tuple(data.get("base_velocity", (0.0, 0.0))),
            arm_action=np.array(data["arm_action"]) if data.get("arm_action") else None,
            gripper_action=data.get("gripper_action", 0.5),
            done=data.get("done", False),
            confidence=data.get("confidence", 1.0),
            control_mode=data.get("control_mode", self._control_mode),
            joint_velocities=np.array(data["joint_velocities"]) if data.get("joint_velocities") else None,
            joint_positions=np.array(data["joint_positions"]) if data.get("joint_positions") else None,
            reasoning=f"[{self._model_type.upper()}_SERVER]"
        )

    def predict_batch(self, observations: List[VLAObservation]) -> List[VLAAction]:
        import requests
        import base64
        from io import BytesIO
        
        if not observations:
            return []
        
        start = time.perf_counter()
        
        batch = []
        for obs in observations:
            image = obs.rgb_image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            pil_img = Image.fromarray(image)
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            batch.append({
                "image": img_b64,
                "instruction": obs.instruction,
                "proprioception": obs.proprioception.tolist()
            })
        
        response = requests.post(self._batch_url, json={"batch": batch}, timeout=30)
        response.raise_for_status()
        results_data = response.json().get("results", [])
        
        self._last_metrics = VLAMetrics(latency_ms=(time.perf_counter() - start) * 1000)
        
        actions = []
        for data in results_data:
            actions.append(VLAAction(
                base_velocity=tuple(data.get("base_velocity", (0.0, 0.0))),
                arm_action=np.array(data["arm_action"]) if data.get("arm_action") else None,
                gripper_action=data.get("gripper_action", 0.5),
                done=data.get("done", False),
                confidence=data.get("confidence", 1.0),
                control_mode=data.get("control_mode", self._control_mode),
                joint_velocities=np.array(data["joint_velocities"]) if data.get("joint_velocities") else None,
                reasoning=f"[{self._model_type.upper()}_SERVER]"
            ))
        
        return actions

    def get_metrics(self) -> VLAMetrics:
        return self._last_metrics

    def reset(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return f"LeRobotServer({self._model_type})"

    @property
    def latency_budget_ms(self) -> float:
        return self._latency_budget_ms
