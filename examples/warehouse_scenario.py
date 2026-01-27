#!/usr/bin/env python3
"""
Warehouse Transport Scenario

Full realistic simulation:
1. TDW physics backend (real physics, real sensors)
2. OpenVLA model (real inference, real latency)
3. Multi-robot coordination (DSM + Fleet)
4. Workload profiling for R3 hardware sizing

Requirements:
    pip install tdw magnebot transformers torch bitsandbytes
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Parse args early to set logging level
parser = argparse.ArgumentParser(description="PhysicAI Warehouse Simulation")
parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug output")
parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps")
parser.add_argument("--resolution", type=str, default="720p", 
                    choices=["480p", "720p", "1080p"],
                    help="Video recording resolution (default: 720p)")
parser.add_argument("--vla", type=str, default="cogact",
                    choices=["cogact", "cogact_server", "openvla", "profiled"],
                    help="VLA model to use (default: cogact)")
parser.add_argument("--instruction", type=str, default="pick up the box and move it forward",
                    help="Task instruction for VLA (default: pick up the box)")
args = parser.parse_args()

# Resolution presets
RESOLUTIONS = {
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080)
}

# Initialize CUDA BEFORE any TDW imports to prevent Unity from blocking GPU access
import torch
if torch.cuda.is_available():
    print(f"CUDA initialized: {torch.cuda.device_count()} GPUs available")
    # Force CUDA initialization
    torch.cuda.init()
    _ = torch.zeros(1).cuda()
else:
    print("WARNING: CUDA not available - VLA will run on CPU (very slow)")

import logging
log_level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from simulator.core import Simulator, SimulatorConfig


def run_simulation():
    """Run realistic warehouse simulation"""
    print("=" * 60)
    print("PhysicAI Warehouse Simulation")
    print("TDW Physics + OpenVLA + DSM Coordination")
    print("=" * 60)
    
    config = SimulatorConfig(
        n_robots=3,
        
        vla_model=args.vla,
        vla_latency_budget_ms=100.0,
        vla_quantization="none",  # Use FP16 - A6000 has plenty of VRAM
        
        enable_dsm=True,
        gossip_period_ms=50.0,
        enable_lf_coordination=True,
        
        max_steps=args.steps,
        profiling_enabled=True,
        
        # Enable video recording (set to True to record)
        enable_recording=True,
        recording_path="recordings/warehouse_sim",
        recording_resolution=RESOLUTIONS[args.resolution],
        
        # Demo instruction - CogACT is trained on manipulation tasks
        # Good prompts: "pick up the object", "move to the target", "push the box"
        demo_instruction=args.instruction,
        inference_interval=5,
        
        # Verbose mode
        verbose=args.verbose
    )
    
    print(f"\nConfiguration:")
    print(f"  Robots: {config.n_robots}")
    print(f"  VLA Model: {config.vla_model} ({config.vla_quantization})")
    print(f"  Instruction: \"{config.demo_instruction}\"")
    print(f"  VLA Latency Budget: {config.vla_latency_budget_ms}ms")
    print(f"  DSM Gossip Period: {config.gossip_period_ms}ms")
    print(f"  Video Recording: {'ON' if config.enable_recording else 'OFF'}")
    if config.enable_recording:
        print(f"  Resolution: {config.recording_resolution[0]}x{config.recording_resolution[1]} ({args.resolution})")
    
    sim = Simulator(config)
    
    print("\nInitializing (this may take time to load VLA model)...")
    if not sim.initialize():
        print("ERROR: Failed to initialize simulator")
        print("Make sure TDW and OpenVLA dependencies are installed:")
        print("  pip install tdw magnebot transformers torch bitsandbytes")
        return
    
    # Add objects to the scene for more interesting tasks
    print("\nSpawning objects in warehouse...")
    objects = [
        ("box_1", "box", (0.0, 0.5, 0.0)),      # Center
        ("box_2", "box", (2.0, 0.5, 2.0)),      # Near robot_1
        ("box_3", "box", (-2.0, 0.5, -2.0)),    # Near robot_0
        ("crate_1", "crate", (4.0, 0.5, 0.0)),  # East
        ("crate_2", "crate", (-4.0, 0.5, 0.0)), # West
        ("bin_1", "bin", (0.0, 0.5, 4.0)),      # North
    ]
    for obj_id, obj_type, pos in objects:
        try:
            sim._backend.spawn_object(obj_id, obj_type, pos)
        except Exception as e:
            print(f"  Warning: Could not spawn {obj_id}: {e}")
    
    # Track initial positions for movement detection (capture NOW before sim starts)
    initial_positions = {}
    for rid, robot in sim.robots.items():
        initial_positions[rid] = robot._position
    
    def progress_callback(state):
        if state.step_count % 100 == 0:
            rtf = state.sim_time / max(state.real_time, 0.001)
            print(f"  Step {state.step_count}: sim={state.sim_time:.2f}s, RTF={rtf:.2f}x")
            
            # In verbose mode, show position changes
            if args.verbose:
                for rid, robot in sim.robots.items():
                    pos = robot._position
                    init = initial_positions[rid]
                    dx = pos[0] - init[0]
                    dz = pos[2] - init[2]
                    dist = (dx**2 + dz**2) ** 0.5
                    print(f"    {rid}: pos=({pos[0]:.2f}, {pos[2]:.2f}) moved={dist:.3f}m from start")
    
    print("\nRunning simulation...")
    sim.run(callback=progress_callback)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    coord_metrics = sim.coordinator.get_metrics()
    print(f"\nCoordination (R1):")
    print(f"  Tasks created: {coord_metrics['total_tasks_created']}")
    print(f"  Tasks completed: {coord_metrics['tasks_completed']}")
    print(f"  Throughput: {coord_metrics['throughput_per_hour']:.1f} tasks/hour")
    
    dsm_metrics = sim.dsm.get_metrics()
    print(f"\nDSM (R1):")
    print(f"  Agents: {dsm_metrics['n_agents']}")
    print(f"  Gossip rounds: {dsm_metrics['gossip_rounds']}")
    print(f"  Updates/round: {dsm_metrics['avg_updates_per_round']:.1f}")
    
    workload = sim.get_metrics()
    if workload:
        print(f"\nVLA Workload (R2/R3):")
        print(f"  Total inferences: {workload.vla_inferences}")
        print(f"  Avg latency: {workload.vla_avg_latency_ms:.1f}ms")
        print(f"  Max latency: {workload.vla_max_latency_ms:.1f}ms")
        print(f"  Contract violations: {workload.vla_violations}")
        print(f"  Sensor bandwidth: {workload.sensor_bandwidth_mbps:.1f} Mbps")
    
    print("\n" + "=" * 60)
    print("HARDWARE RECOMMENDATIONS (R3)")
    print("=" * 60)
    
    if sim._profiler:
        recommendations = sim._profiler.get_hardware_recommendations()
        
        print(f"\nCompute (uAgent SoC):")
        print(f"  NPU: {recommendations['compute']['npu_tops']:.0f} TOPS")
        print(f"  VLA rate: {recommendations['compute']['vla_inference_rate_hz']:.1f} Hz")
        print(f"  Platform: {recommendations['compute']['recommendation']}")
        
        print(f"\nMemory:")
        print(f"  VLA weights: {recommendations['memory']['vla_weights_gb']} GB")
        print(f"  Context buffer: {recommendations['memory']['context_buffer_mb']} MB")
        
        print(f"\nCommunication:")
        print(f"  Sensor: {recommendations['communication']['sensor_bandwidth_mbps']:.1f} Mbps")
        print(f"  Gossip: {recommendations['communication']['gossip_bandwidth_kbps']} Kbps")
        
        print(f"\nContracts: {'MET' if recommendations['contracts_met'] else 'VIOLATED'}")
        
        output_path = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(output_path, exist_ok=True)
        sim._profiler.export(os.path.join(output_path, 'workload_profile.json'))
        print(f"\nProfile saved to: results/workload_profile.json")
    
    sim.close()
    print("\nSimulation complete.")


if __name__ == "__main__":
    run_simulation()
