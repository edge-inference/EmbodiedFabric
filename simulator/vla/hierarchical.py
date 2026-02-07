"""Hierarchical VLA: Routes to NAV or MANIP experts via VLM planner."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import time

from .interface import VLAInterface, VLAObservation, VLAAction, VLAMetrics


class TaskMode(Enum):
    NAVIGATION = "nav"
    MANIPULATION = "manip"
    IDLE = "idle"


@dataclass
class Subgoal:
    mode: TaskMode
    target: Optional[str] = None
    position: Optional[Tuple[float, float, float]] = None
    instruction: Optional[str] = None
    priority: float = 1.0


class HierarchicalPlanner(VLAInterface):

    def __init__(self,
                 nav_expert: Optional[VLAInterface] = None,
                 manip_expert: Optional[VLAInterface] = None,
                 vlm_planner_url: Optional[str] = None,
                 planner_freq_hz: float = 2.0,
                 latency_budget_ms: float = 150.0,
                 device: str = "cuda:0"):
        
        self._nav_expert = nav_expert
        self._manip_expert = manip_expert
        self._vlm_planner_url = vlm_planner_url
        self._planner_freq_hz = planner_freq_hz
        self._latency_budget_ms = latency_budget_ms
        self._device = device
        
        self._vlm_planner = None
        self._current_mode = TaskMode.IDLE
        self._current_subgoal: Optional[Subgoal] = None
        self._instruction_queue: List[str] = []
        self._last_plan_time = 0.0
        self._plan_interval = 1.0 / planner_freq_hz
        self._last_metrics = VLAMetrics()

    def _init_experts(self):
        if self._nav_expert is None:
            from .nomad import NoMaDNavigator
            self._nav_expert = NoMaDNavigator(device=self._device)
        
        if self._manip_expert is None:
            try:
                from .cogact_server import CogACTServerClient
                self._manip_expert = CogACTServerClient()
            except Exception:
                from .cogact import CogACTVLA
                self._manip_expert = CogACTVLA(device=self._device)
        
        if self._vlm_planner is None:
            from .vlm_planner import VLMPlanner
            self._vlm_planner = VLMPlanner(server_url=self._vlm_planner_url, device=self._device)

    def _run_vlm_planner(self, observation: VLAObservation) -> Optional[Subgoal]:
        current_time = time.time()
        if current_time - self._last_plan_time < self._plan_interval:
            return self._current_subgoal
        
        self._last_plan_time = current_time
        
        if observation.instruction and observation.instruction not in self._instruction_queue:
            self._instruction_queue = [observation.instruction]
            
            plan = self._vlm_planner.plan(observation)
            
            if plan.mode.value == "nav":
                mode = TaskMode.NAVIGATION
            else:
                mode = TaskMode.MANIPULATION
            
            self._current_subgoal = Subgoal(
                mode=mode,
                target=plan.target_object or plan.target_location,
                instruction=plan.subgoal or observation.instruction
            )
            self._current_mode = mode
        
        return self._current_subgoal

    def predict(self, observation: VLAObservation) -> VLAAction:
        self._init_experts()
        start = time.perf_counter()
        
        subgoal = self._run_vlm_planner(observation)
        
        if self._current_mode == TaskMode.NAVIGATION:
            expert = self._nav_expert
            expert_name = "NAV"
        else:
            expert = self._manip_expert
            expert_name = "MANIP"
        
        action = expert.predict(observation)
        
        expert_metrics = expert.get_metrics()
        self._last_metrics = VLAMetrics(
            latency_ms=(time.perf_counter() - start) * 1000,
            flops=expert_metrics.flops,
            memory_bytes=expert_metrics.memory_bytes,
            tokens_processed=expert_metrics.tokens_processed
        )
        
        action.reasoning = f"[{expert_name}] {subgoal.instruction if subgoal else observation.instruction}"
        return action

    def predict_batch(self, observations: List[VLAObservation]) -> List[VLAAction]:
        self._init_experts()
        
        nav_obs, manip_obs = [], []
        nav_indices, manip_indices = [], []
        
        for i, obs in enumerate(observations):
            subgoal = self._run_vlm_planner(obs)
            if self._current_mode == TaskMode.NAVIGATION:
                nav_obs.append(obs)
                nav_indices.append(i)
            else:
                manip_obs.append(obs)
                manip_indices.append(i)
        
        results = [None] * len(observations)
        
        if nav_obs:
            if hasattr(self._nav_expert, 'predict_batch'):
                nav_actions = self._nav_expert.predict_batch(nav_obs)
            else:
                nav_actions = [self._nav_expert.predict(obs) for obs in nav_obs]
            for idx, action in zip(nav_indices, nav_actions):
                action.reasoning = "[NAV]"
                results[idx] = action
        
        if manip_obs:
            if hasattr(self._manip_expert, 'predict_batch'):
                manip_actions = self._manip_expert.predict_batch(manip_obs)
            else:
                manip_actions = [self._manip_expert.predict(obs) for obs in manip_obs]
            for idx, action in zip(manip_indices, manip_actions):
                action.reasoning = "[MANIP]"
                results[idx] = action
        
        return results

    def get_metrics(self) -> VLAMetrics:
        return self._last_metrics

    def reset(self) -> None:
        """Reset planner and expert states."""
        self._current_mode = TaskMode.IDLE
        self._current_subgoal = None
        self._instruction_queue.clear()
        
        if self._nav_expert:
            self._nav_expert.reset()
        if self._manip_expert:
            self._manip_expert.reset()

    @property
    def model_name(self) -> str:
        return f"Hierarchical(NAV={self._nav_expert.model_name if self._nav_expert else 'None'}, MANIP={self._manip_expert.model_name if self._manip_expert else 'None'})"

    @property
    def latency_budget_ms(self) -> float:
        return self._latency_budget_ms

    @property
    def current_mode(self) -> TaskMode:
        return self._current_mode

    @property
    def current_subgoal(self) -> Optional[Subgoal]:
        return self._current_subgoal
