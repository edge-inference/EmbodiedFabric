"""Simulator orchestration for multi-robot VLA experiments."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class SimulatorConfig:
    n_robots: int = 5
    
    scene_id: str = "warehouse_default"
    scene_name: Optional[str] = None              
    scene_size: tuple = (20, 20)                  
    floorplan_layout: Optional[int] = None
    spawn_positions: Optional[List[Tuple[float, float, float]]] = None
    
    vla_model: str = "smolvla"
    vla_latency_budget_ms: float = 100.0
    vla_quantization: str = "4bit"
    
    enable_dsm: bool = True
    gossip_period_ms: float = 50.0
    
    enable_lf_coordination: bool = True
    
    time_step: float = 0.02
    max_steps: int = 10000
    
    seed: Optional[int] = None
    
    profiling_enabled: bool = True
    
    inference_interval: int = 1
    demo_instruction: Optional[str] = None

    enable_recording: bool = False
    recording_path: str = "recordings"
    recording_resolution: tuple = (1280, 720)
    recording_view: str = "overhead"

    backend: str = "isaac"

    # Debug
    verbose: bool = False                         


@dataclass
class SimulatorState:
    """Current simulator state"""
    step_count: int = 0
    sim_time: float = 0.0
    real_time: float = 0.0
    robots: Dict[str, Any] = field(default_factory=dict)
    tasks: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


class Simulator:
    """Batched multi-robot loop: obs -> VLA -> actions -> metrics."""
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.state = SimulatorState()
        
        self._backend = None
        self._robots: Dict[str, 'VLAAgent'] = {}
        self._coordinator = None
        self._dsm = None
        self._profiler = None
        
        self._initialized = False
    
    def initialize(self) -> bool:
        logger.info(f"Initializing simulator: robots={self.config.n_robots}, "
                   f"vla={self.config.vla_model}, backend={self.config.backend}")
        
        from .backend.isaac_backend import IsaacSimBackend
        self._backend = IsaacSimBackend(
            self.config,
            enable_recording=self.config.enable_recording,
            recording_path=self.config.recording_path
        )
        
        if not self._backend.initialize():
            logger.error(f"Failed to initialize {self.config.backend} backend")
            return False
        
        self._create_robots()
        
        if self.config.enable_dsm:
            from .coordination.dsm import DistributedSharedMemory
            self._dsm = DistributedSharedMemory(
                n_agents=self.config.n_robots,
                gossip_period_ms=self.config.gossip_period_ms
            )
        
        from .coordination.fleet import FleetCoordinator
        self._coordinator = FleetCoordinator(
            n_robots=self.config.n_robots,
            dsm=self._dsm,
            enable_lf=self.config.enable_lf_coordination
        )
        
        if self.config.profiling_enabled:
            try:
                from workload.profiler import WorkloadProfiler
            except ImportError:
                from ..workload.profiler import WorkloadProfiler
            self._profiler = WorkloadProfiler()
        
        self._initialized = True
        logger.info("Simulator initialized successfully")
        return True
    
    def _create_robots(self):
        """Spawn robots and wrap them with VLAAgent."""
        from .robot.vla_agent import VLAAgent
        import math
        
        grid_size = math.ceil(math.sqrt(self.config.n_robots))
        spacing = min(self.config.scene_size) / (grid_size + 1)

        spawn_positions = list(self.config.spawn_positions or [])
        
        for i in range(self.config.n_robots):
            robot_id = f"robot_{i}"
            
            if i < len(spawn_positions):
                position = spawn_positions[i]
            else:
                row = i // grid_size
                col = i % grid_size
                x = (col + 1) * spacing - self.config.scene_size[0] / 2
                z = (row + 1) * spacing - self.config.scene_size[1] / 2
                position = (x, 0.0, z)
            
            if not self._backend.spawn_robot(robot_id, position):
                logger.error("Failed to spawn robot %s", robot_id)
                continue
            
            robot = VLAAgent(
                robot_id=robot_id,
                backend=self._backend,
                vla_model=self.config.vla_model,
                vla_latency_budget_ms=self.config.vla_latency_budget_ms,
                vla_quantization=self.config.vla_quantization,
                inference_interval=self.config.inference_interval,
                demo_instruction=self.config.demo_instruction
            )
            
            self._robots[robot_id] = robot
        
        logger.info("Created %d VLA-driven robots", len(self._robots))
    
    def step(self) -> SimulatorState:
        if not self._initialized:
            raise RuntimeError("Simulator not initialized")
        
        step_start = time.perf_counter()
        
        self._backend.step()
        
        self._coordinator.step()
        
        batch_obs = []
        batch_agents = []
        for robot in self._robots.values():
            bundle = robot.build_vla_observation(self._coordinator, self._dsm)
            if bundle is None:
                continue
            backend_obs, vla_obs = bundle
            batch_obs.append(vla_obs)
            batch_agents.append((robot, backend_obs, vla_obs))
        
        if batch_obs:
            vla_model = batch_agents[0][0].vla
            if hasattr(vla_model, "predict_batch"):
                actions = vla_model.predict_batch(batch_obs)
            else:
                actions = [vla_model.predict(obs) for obs in batch_obs]
            
            for (robot, backend_obs, vla_obs), action in zip(batch_agents, actions):
                robot.apply_vla_action(
                    action, backend_obs, vla_obs, self._profiler, self._dsm
                )
        
        if self._dsm:
            self._dsm.gossip_round(list(self._robots.values()))
        
        self.state.step_count += 1
        self.state.sim_time += self.config.time_step
        self.state.real_time += time.perf_counter() - step_start
        
        if self._profiler:
            self._profiler.record_step(self.state)
        
        return self.state
    
    def run(self, steps: Optional[int] = None, callback=None) -> SimulatorState:
        max_steps = steps or self.config.max_steps
        
        logger.info(f"Running simulation for {max_steps} steps")
        
        if self._profiler:
            self._profiler.start_recording()
        
        for step in range(max_steps):
            self.step()
            
            if callback:
                callback(self.state)
            
            if step % 100 == 0:
                rtf = self.state.sim_time / max(self.state.real_time, 0.001)
                logger.info(f"Step {step}: sim_time={self.state.sim_time:.2f}s, "
                           f"RTF={rtf:.2f}x")
            
            if self.config.verbose and step % 50 == 0:
                positions = []
                for rid, robot in self._robots.items():
                    pos = robot.position
                    positions.append(f"{rid}=({pos[0]:.2f},{pos[2]:.2f})")
                logger.debug(f"[Step {step}] Positions: {', '.join(positions)}")
        
        if self._profiler:
            self._profiler.stop_recording()
        
        return self.state
    
    def get_metrics(self):
        if self._profiler:
            return self._profiler.get_metrics()
        return None
    
    def close(self):
        if self._backend:
            self._backend.close()
        self._initialized = False
        logger.info("Simulator closed")
    
    @property
    def robots(self) -> Dict[str, 'VLAAgent']:
        return self._robots
    
    @property
    def coordinator(self):
        return self._coordinator
    
    @property
    def dsm(self):
        return self._dsm
