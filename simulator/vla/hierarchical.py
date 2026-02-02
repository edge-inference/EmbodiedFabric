"""
Hierarchical VLA Planner

Routes instructions to NAV (NoMaD) or MANIP (CogACT) experts.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import time
import re
import numpy as np

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
    """Routes to NAV (NoMaD) or MANIP (CogACT) based on instruction."""

    def __init__(self,
                 nav_expert: Optional[VLAInterface] = None,
                 manip_expert: Optional[VLAInterface] = None,
                 planner_model: str = "rule_based",
                 planner_freq_hz: float = 2.0,
                 latency_budget_ms: float = 150.0,
                 device: str = "cuda:0"):
        
        self._nav_expert = nav_expert
        self._manip_expert = manip_expert
        self._planner_model = planner_model
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
        self._nav_keywords = [
            "go", "move", "navigate", "drive", "travel", "walk", "head",
            "approach", "reach", "find", "locate", "search", "explore"
        ]
        self._manip_keywords = [
            "pick", "grasp", "grab", "lift", "place", "put", "drop",
            "push", "pull", "open", "close", "turn", "rotate", "press"
        ]

    def _init_experts(self):
        """Lazy initialization of expert models."""
        if self._nav_expert is None:
            try:
                from .nomad import NoMaDNavigator
                self._nav_expert = NoMaDNavigator(device=self._device)
            except ImportError:
                from .nomad import MockNoMaDNavigator
                self._nav_expert = MockNoMaDNavigator()
        
        if self._manip_expert is None:
            try:
                from .cogact_server import CogACTServerClient
                self._manip_expert = CogACTServerClient()
            except ImportError:
                from .cogact import CogACTVLA
                self._manip_expert = CogACTVLA(device=self._device)

    def _classify_instruction(self, instruction: str) -> TaskMode:
        """
        Rule-based instruction classification.
        Returns NAV or MANIP based on keyword matching.
        """
        instruction_lower = instruction.lower()
        
        nav_score = sum(1 for kw in self._nav_keywords if kw in instruction_lower)
        manip_score = sum(1 for kw in self._manip_keywords if kw in instruction_lower)
        
        if manip_score > nav_score:
            return TaskMode.MANIPULATION
        elif nav_score > 0:
            return TaskMode.NAVIGATION
        else:
            return TaskMode.MANIPULATION

    def _parse_instruction(self, instruction: str) -> List[Subgoal]:
        """
        Parse instruction into subgoals.
        
        Example: "go to shelf A3, pick box, return to dock"
        -> [NAV(shelf_A3), MANIP(pick_box), NAV(dock)]
        """
        subgoals = []
        
        parts = re.split(r'[,;]|\band\b|\bthen\b', instruction.lower())
        parts = [p.strip() for p in parts if p.strip()]
        
        for part in parts:
            mode = self._classify_instruction(part)
            
            target = None
            location_patterns = [
                r'to\s+(?:the\s+)?(\w+(?:\s+\w+)?)',
                r'at\s+(?:the\s+)?(\w+(?:\s+\w+)?)',
                r'(?:shelf|station|dock|bin)\s*(\w+)',
            ]
            for pattern in location_patterns:
                match = re.search(pattern, part)
                if match:
                    target = match.group(1)
                    break
            
            subgoals.append(Subgoal(
                mode=mode,
                target=target,
                instruction=part.strip()
            ))
        
        if not subgoals:
            mode = self._classify_instruction(instruction)
            subgoals.append(Subgoal(
                mode=mode,
                instruction=instruction
            ))
        
        return subgoals

    def _run_vlm_planner(self, observation: VLAObservation) -> Optional[Subgoal]:
        """
        Run high-level VLM planner to decide mode and subgoal.
        
        For now uses rule-based parsing. to be replaced with actual VLM
        """
        current_time = time.time()
        if current_time - self._last_plan_time < self._plan_interval:
            return self._current_subgoal
        
        self._last_plan_time = current_time
        
        if observation.instruction and observation.instruction not in self._instruction_queue:
            self._instruction_queue = [observation.instruction]
            subgoals = self._parse_instruction(observation.instruction)
            if subgoals:
                self._current_subgoal = subgoals[0]
                self._current_mode = self._current_subgoal.mode
        
        return self._current_subgoal

    def predict(self, observation: VLAObservation) -> VLAAction:
        """
        Hierarchical prediction:
        1. High-level planner decides mode
        2. Route to appropriate expert
        3. Return expert's action
        """
        self._init_experts()
        start = time.perf_counter()
        
        subgoal = self._run_vlm_planner(observation)
        
        if subgoal is None or self._current_mode == TaskMode.IDLE:
            self._current_mode = self._classify_instruction(observation.instruction)
        
        if self._current_mode == TaskMode.NAVIGATION:
            expert = self._nav_expert
            expert_name = "NAV"
        else:
            expert = self._manip_expert
            expert_name = "MANIP"
        
        action = expert.predict(observation)
        
        expert_metrics = expert.get_metrics()
        total_latency = (time.perf_counter() - start) * 1000
        
        self._last_metrics = VLAMetrics(
            latency_ms=total_latency,
            flops=expert_metrics.flops,
            memory_bytes=expert_metrics.memory_bytes,
            tokens_processed=expert_metrics.tokens_processed
        )
        
        action.reasoning = f"[{expert_name}] {subgoal.instruction if subgoal else observation.instruction}"
        
        return action

    def predict_batch(self, observations: List[VLAObservation]) -> List[VLAAction]:
        """Batch prediction - routes each observation to appropriate expert."""
        self._init_experts()
        
        nav_obs = []
        manip_obs = []
        nav_indices = []
        manip_indices = []
        
        for i, obs in enumerate(observations):
            mode = self._classify_instruction(obs.instruction)
            if mode == TaskMode.NAVIGATION:
                nav_obs.append(obs)
                nav_indices.append(i)
            else:
                manip_obs.append(obs)
                manip_indices.append(i)
        
        results = [None] * len(observations)
        
        if nav_obs and hasattr(self._nav_expert, 'predict_batch'):
            nav_actions = self._nav_expert.predict_batch(nav_obs)
            for idx, action in zip(nav_indices, nav_actions):
                action.reasoning = "[NAV]"
                results[idx] = action
        elif nav_obs:
            for idx, obs in zip(nav_indices, nav_obs):
                action = self._nav_expert.predict(obs)
                action.reasoning = "[NAV]"
                results[idx] = action
        
        if manip_obs and hasattr(self._manip_expert, 'predict_batch'):
            manip_actions = self._manip_expert.predict_batch(manip_obs)
            for idx, action in zip(manip_indices, manip_actions):
                action.reasoning = "[MANIP]"
                results[idx] = action
        elif manip_obs:
            for idx, obs in zip(manip_indices, manip_obs):
                action = self._manip_expert.predict(obs)
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
