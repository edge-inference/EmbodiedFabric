"""
Simulator Core

Main entry point for the unified simulator.
Orchestrates TDW physics backend, VLA robots, and coordination.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class SimulatorConfig:
    """Simulator configuration"""
    
    n_robots: int = 5
    
    scene_id: str = "warehouse_default"
    scene_size: tuple = (20, 20)                  # meters
    
    vla_model: str = "openvla"                    # "openvla" or "profiled"
    vla_latency_budget_ms: float = 100.0          # Realistic budget for 7B model
    vla_quantization: str = "4bit"                # For BlockDialect R3
    
    enable_dsm: bool = True                       # Distributed shared memory
    gossip_period_ms: float = 50.0
    
    enable_lf_coordination: bool = True           # Lingua Franca control plane
    
    time_step: float = 0.02                       # 50Hz physics
    max_steps: int = 10000
    
    seed: Optional[int] = None
    
    profiling_enabled: bool = True
    
    # VLA control
    inference_interval: int = 1                 # Run VLA every N steps
    demo_instruction: Optional[str] = None      # Override instruction (demo)
    
    # Video recording
    enable_recording: bool = False                # Record video of simulation
    recording_path: str = "recordings"            # Path to save recordings
    recording_resolution: tuple = (1280, 720)     # Video resolution (width, height)
    
    # Debug
    verbose: bool = False                         # Enable verbose debug output


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
    """
    Unified simulator for PhysicAI research.
    
    Integrates:
    - TDW physics backend (real physics)
    - OpenVLA inference (real VLA)
    - Multi-robot coordination (DSM + LF)
    - Workload profiling for hardware sizing
    """
    
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
        """Initialize all simulator components"""
        logger.info(f"Initializing simulator: robots={self.config.n_robots}, "
                   f"vla={self.config.vla_model}")
        
        from .backend.tdw_backend import TDWBackend
        self._backend = TDWBackend(
            self.config,
            enable_recording=self.config.enable_recording,
            recording_path=self.config.recording_path
        )
        if not self._backend.initialize():
            logger.error("Failed to initialize TDW backend")
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
        """Create VLA-driven robot instances and spawn them in TDW"""
        from .robot.vla_agent import VLAAgent
        import math
        
        # Calculate spawn positions in a grid pattern
        grid_size = math.ceil(math.sqrt(self.config.n_robots))
        spacing = min(self.config.scene_size) / (grid_size + 1)
        
        for i in range(self.config.n_robots):
            robot_id = f"robot_{i}"
            
            # Grid position (centered in scene)
            row = i // grid_size
            col = i % grid_size
            x = (col + 1) * spacing - self.config.scene_size[0] / 2
            z = (row + 1) * spacing - self.config.scene_size[1] / 2
            position = (x, 0.0, z)
            
            # Spawn in TDW first
            if not self._backend.spawn_robot(robot_id, position):
                logger.error(f"Failed to spawn robot {robot_id} in TDW")
                continue
            
            # Create VLA agent wrapper
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
        
        logger.info(f"Created {len(self._robots)} VLA-driven robots in TDW")
    
    def step(self) -> SimulatorState:
        """Execute one simulation step"""
        if not self._initialized:
            raise RuntimeError("Simulator not initialized")
        
        step_start = time.perf_counter()
        
        # Advance physics first so sensors are updated for this step
        self._backend.step()
        
        self._coordinator.step()
        
        # Batch inference path for CogACT (local or server)
        if self.config.vla_model in ("cogact", "cogact_server"):
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
                vla_model = batch_agents[0][0]._vla
                if hasattr(vla_model, "predict_batch"):
                    actions = vla_model.predict_batch(batch_obs)
                else:
                    actions = [vla_model.predict(obs) for obs in batch_obs]
                
                for (robot, backend_obs, vla_obs), action in zip(batch_agents, actions):
                    robot.apply_vla_action(action, backend_obs, vla_obs, self._profiler, self._dsm)
        else:
            for robot in self._robots.values():
                robot.step(
                    coordinator=self._coordinator,
                    dsm=self._dsm,
                    profiler=self._profiler
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
        """Run simulation for specified steps"""
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
            
            # Verbose: log robot positions every 50 steps
            if self.config.verbose and step % 50 == 0:
                positions = []
                for rid, robot in self._robots.items():
                    pos = robot._position
                    positions.append(f"{rid}=({pos[0]:.2f},{pos[2]:.2f})")
                logger.debug(f"[Step {step}] Positions: {', '.join(positions)}")
        
        if self._profiler:
            self._profiler.stop_recording()
        
        return self.state
    
    def get_metrics(self):
        """Get profiled metrics"""
        if self._profiler:
            return self._profiler.get_metrics()
        return None
    
    def close(self):
        """Cleanup simulator resources"""
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
