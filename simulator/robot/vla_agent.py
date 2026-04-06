"""Robot wrapper that runs a VLA model and emits backend commands."""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import logging

from .base import RobotAgent, RobotState, RobotStatus
from ..vla.interface import VLAObservation, create_vla
from ..backend.base import RobotCommand

logger = logging.getLogger(__name__)


class VLAAgent(RobotAgent):
    """Runs VLA inference and applies the action each step."""
    
    def __init__(self,
                 robot_id: str,
                 backend,
                 vla_model: str = "smolvla",
                 vla_latency_budget_ms: float = 100.0,
                 vla_quantization: str = "4bit",
                 inference_interval: int = 1,
                 demo_instruction: Optional[str] = None):
        super().__init__(robot_id, backend)
        
        self._vla = create_vla(
            vla_model,
            latency_budget_ms=vla_latency_budget_ms,
            quantization=vla_quantization
        )
        self._vla_latency_budget_ms = vla_latency_budget_ms
        self._inference_interval = max(1, inference_interval)
        self._step_counter = 0
        
        self._current_instruction = demo_instruction or "Wait for task"
        self._observation_history = []
        self._max_history = 5
        
        self._position = (0.0, 0.0, 0.0)
        self._last_action = None
        self._waiting_for_action = False
    
    def step(self, coordinator, dsm, profiler) -> None:
        bundle = self.build_vla_observation(coordinator, dsm)
        if bundle is None:
            return
        backend_obs, vla_obs = bundle
        
        action = self._vla.predict(vla_obs)
        self._last_action = action
        
        self.apply_vla_action(action, backend_obs, vla_obs, profiler, dsm)

    def build_vla_observation(self, coordinator, dsm):
        self._metrics['total_steps'] += 1
        self._step_counter += 1
        
        if self._waiting_for_action:
            if hasattr(self._backend, 'is_robot_idle') and not self._backend.is_robot_idle(self.robot_id):
                return None
            self._waiting_for_action = False
        
        if (self._step_counter % self._inference_interval) != 0:
            return None
        
        try:
            backend_obs = self._backend.get_observation(self.robot_id)
            self._position = backend_obs.position
        except Exception as e:
            logger.warning("Failed to get observation for %s: %s", self.robot_id, e)
            return None
        
        context = self._build_context(coordinator, dsm)
        
        vla_obs = VLAObservation(
            rgb_image=backend_obs.rgb,
            depth_image=backend_obs.depth,
            instruction=self._current_instruction,
            proprioception=np.array([
                *backend_obs.position,
                *backend_obs.rotation,
                float(backend_obs.gripper_state)
            ]),
            context=context,
            history=self._observation_history[-self._max_history:]
        )
        return backend_obs, vla_obs

    def apply_vla_action(self, action, backend_obs, vla_obs, profiler, dsm) -> None:
        vla_metrics = self._vla.get_metrics()
        if profiler:
            profiler.record_compute_event(
                event_type="vla_inference",
                flops=vla_metrics.flops,
                duration_ms=vla_metrics.latency_ms
            )
            if vla_metrics.latency_ms > self._vla_latency_budget_ms:
                profiler.record_contract_violation(
                    contract="vla_timing",
                    actual=vla_metrics.latency_ms,
                    budget=self._vla_latency_budget_ms
                )
        
        logger.debug(
            "[%s] VLA action base=(%.3f, %.3f) grip=%.2f done=%s pos=(%.2f, %.2f)",
            self.robot_id,
            action.base_velocity[0],
            action.base_velocity[1],
            action.gripper_action,
            action.done,
            backend_obs.position[0],
            backend_obs.position[2],
        )
        
        control_mode = getattr(action, 'control_mode', 'high_level')
        joint_vels = getattr(action, 'joint_velocities', None)
        joint_pos = getattr(action, 'joint_positions', None)
        
        command = RobotCommand(
            robot_id=self.robot_id,
            linear_velocity=(action.base_velocity[0], 0.0, action.base_velocity[1]),
            angular_velocity=(0.0, 0.0, action.base_velocity[1] * 0.5),
            gripper_action=action.gripper_action,
            arm_target=action.arm_action,
            control_mode=control_mode,
            joint_velocities=joint_vels,
            joint_positions=joint_pos
        )
        
        if self._backend.send_command(command):
            self._waiting_for_action = True
            logger.debug("[%s] Command sent", self.robot_id)
        
        if dsm:
            dsm.write_agent_state(
                agent_id=self.robot_id,
                position=self._position,
                state=self._state.value,
                task_id=self._current_task.get('id') if self._current_task else None
            )
        
        self._update_state(action)
        self._observation_history.append(vla_obs)
        if len(self._observation_history) > self._max_history * 2:
            self._observation_history = self._observation_history[-self._max_history:]
    
    def _build_context(self, coordinator, dsm) -> Dict[str, Any]:
        context = {}
        
        if dsm:
            context['nearby_agents'] = dsm.get_nearby_agents(
                self._position, radius=5.0
            )
            context['jam_signals'] = dsm.get_jam_signals(self._position)
            context['shared_observations'] = dsm.get_shared_observations(
                self._position, radius=10.0
            )
        
        if coordinator:
            context['available_tasks'] = coordinator.get_available_tasks()
            if self._current_task:
                context['target_position'] = self._current_task.get('location')
        
        return context
    
    def _update_state(self, action) -> None:
        if action.done:
            if self._current_task:
                self._metrics['tasks_completed'] += 1
                self._current_task = None
                self._current_instruction = "Wait for task"
            self._state = RobotState.IDLE
        elif action.gripper_action > 0.8 or action.gripper_action < 0.2:
            self._state = RobotState.MANIPULATING
        elif abs(action.base_velocity[0]) > 0.01 or abs(action.base_velocity[1]) > 0.01:
            self._state = RobotState.NAVIGATING
            speed = np.sqrt(action.base_velocity[0]**2 + action.base_velocity[1]**2)
            self._metrics['distance_traveled'] += speed * 0.02
        else:
            self._state = RobotState.WAITING
    
    def assign_task(self, task: Dict[str, Any]) -> bool:
        if self._current_task is not None:
            return False
        
        self._current_task = task
        self._current_instruction = task.get(
            'instruction',
            f"Go to {task.get('location')} and complete task"
        )
        self._state = RobotState.NAVIGATING
        
        logger.info(f"Robot {self.robot_id} assigned task: {self._current_instruction}")
        return True
    
    def get_status(self) -> RobotStatus:
        return RobotStatus(
            robot_id=self.robot_id,
            state=self._state,
            position=self._position,
            current_task_id=self._current_task.get('id') if self._current_task else None
        )
    
    @property
    def vla_metrics(self):
        return self._vla.get_metrics()

    @property
    def vla(self):
        return self._vla

    @property
    def position(self) -> Tuple[float, float, float]:
        return self._position
