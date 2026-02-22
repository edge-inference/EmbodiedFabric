"""
Profiled version of policy_runner.py.
Level 1: Wall-clock timing per stage (always on).
Level 2: torch.profiler on VLA inference steps (--torch_profile flag).
         Also emits NVTX ranges for nsys/ncu scoped capture.
"""
import sys
import io
import time
import statistics
import numpy as np
import random
import torch
import tqdm

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import get_arena_builder_from_cli
from isaaclab_arena.examples.policy_runner_cli import create_policy, setup_policy_argument_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

LOGDIR = "/home/modfi/models/vla_simu/logs/profiling"
TIMINGS = {}


def record(name, ms):
    TIMINGS.setdefault(name, []).append(ms)


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def ms_since(t0):
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def print_report(file=None):
    out = file or sys.stdout
    if not TIMINGS:
        out.write("\n[PROFILER] No timings recorded!\n")
        return
    out.write("\n" + "=" * 75 + "\n")
    out.write(f"  {'STAGE':<35} {'N':>5} {'MEAN':>8} {'MED':>8} {'P95':>8} {'SUM':>9}\n")
    out.write("=" * 75 + "\n")
    for name, vals in TIMINGS.items():
        if not vals:
            continue
        s = sorted(vals)
        p95 = s[min(int(len(s) * 0.95), len(s) - 1)]
        out.write(f"  {name:<35} {len(vals):>5} {statistics.mean(vals):>7.1f}ms "
                  f"{statistics.median(vals):>7.1f}ms {p95:>7.1f}ms {sum(vals):>8.0f}ms\n")
    out.write("=" * 75 + "\n")

    inf = TIMINGS.get("1_get_action_WITH_inference", [])
    replay = TIMINGS.get("2_get_action_chunk_replay", [])
    env = TIMINGS.get("3_env_step", [])
    total = sum(sum(v) for v in [inf, replay, env])
    if total > 0:
        out.write(f"\n  Wall time breakdown ({len(env)} total steps):\n")
        out.write(f"    VLA inference steps ({len(inf):>3}x):  {sum(inf):>8.0f}ms  ({sum(inf)/total*100:>5.1f}%)\n")
        out.write(f"    Chunk replay steps  ({len(replay):>3}x):  {sum(replay):>8.0f}ms  ({sum(replay)/total*100:>5.1f}%)\n")
        out.write(f"    env.step + physics  ({len(env):>3}x):  {sum(env):>8.0f}ms  ({sum(env)/total*100:>5.1f}%)\n")
        out.write(f"    TOTAL:                      {total:>8.0f}ms  ({total/1000:.1f}s)\n")
    out.write("\n")


def main():
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, unknown = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        args_parser = setup_policy_argument_parser(args_parser)
        # Add our profiling flags
        args_parser.add_argument("--torch_profile", action="store_true",
                                 help="Enable torch.profiler on VLA inference calls")
        args_parser.add_argument("--profile_steps", type=int, default=5,
                                 help="Number of VLA inference calls to profile (default: 5)")
        args_cli = args_parser.parse_args()

        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()

        if args_cli.seed is not None:
            env.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            random.seed(args_cli.seed)

        obs, _ = env.reset()
        policy, num_steps = create_policy(args_cli)
        from isaaclab_arena.metrics.metrics import compute_metrics

        num_steps = min(num_steps, 200)
        use_torch_profile = getattr(args_cli, "torch_profile", False)
        max_profile_steps = getattr(args_cli, "profile_steps", 5)

        print(f"\n[PROFILER] Running {num_steps} steps")
        print(f"[PROFILER] torch.profiler: {'ON' if use_torch_profile else 'OFF'}")
        print(f"[PROFILER] NVTX ranges: ON (for nsys/ncu --capture-range=cudaProfilerApi)\n")

        inference_count = 0
        trace_dir = f"{LOGDIR}/torch_traces"
        prof_table_path = f"{LOGDIR}/torch_profile_table.txt"

        # Accumulate profiler results across multiple inference calls
        all_key_averages = []

        for step_i in tqdm.tqdm(range(num_steps)):
            with torch.inference_mode():
                needs_inference = any(policy.env_requires_new_action_chunk)

                if needs_inference:
                    profiling_this_step = use_torch_profile and inference_count < max_profile_steps

                    # NVTX + cudaProfiler scope (for nsys/ncu)
                    torch.cuda.nvtx.range_push("vla_inference")
                    if profiling_this_step:
                        torch.cuda.cudart().cudaProfilerStart()

                    t0 = sync_time()

                    if profiling_this_step:
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            record_shapes=True,
                            profile_memory=True,
                            with_stack=False,
                        ) as prof:
                            actions = policy.get_action(env, obs)

                        elapsed = ms_since(t0)

                        # Export chrome trace for this inference call
                        import os
                        os.makedirs(trace_dir, exist_ok=True)
                        trace_path = f"{trace_dir}/inference_{inference_count}.json"
                        prof.export_chrome_trace(trace_path)

                        all_key_averages.append(
                            prof.key_averages(group_by_input_shape=True)
                        )

                        print(f"\n[PROFILER] Inference #{inference_count} traced -> {trace_path}")

                    else:
                        actions = policy.get_action(env, obs)
                        elapsed = ms_since(t0)

                    if profiling_this_step:
                        torch.cuda.cudart().cudaProfilerStop()
                    torch.cuda.nvtx.range_pop()

                    record("1_get_action_WITH_inference", elapsed)
                    inference_count += 1
                else:
                    t0 = sync_time()
                    actions = policy.get_action(env, obs)
                    record("2_get_action_chunk_replay", ms_since(t0))

                # --- env.step ---
                t0 = sync_time()
                obs, _, terminated, truncated, _ = env.step(actions)
                record("3_env_step", ms_since(t0))

                if terminated.any() or truncated.any():
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)

        # --- Write reports before SimulationAppContext kills us ---
        buf = io.StringIO()
        print_report(file=buf)
        metrics = compute_metrics(env)
        buf.write(f"Metrics: {metrics}\n")

        # Write torch.profiler summary table
        if all_key_averages:
            buf.write("\n" + "=" * 75 + "\n")
            buf.write("  TORCH PROFILER - Top 30 ops by CUDA time (averaged over "
                      f"{len(all_key_averages)} inference calls)\n")
            buf.write("=" * 75 + "\n")
            # Use the last profile's key_averages for the summary table
            table = all_key_averages[-1].table(
                sort_by="cuda_time_total", row_limit=30
            )
            buf.write(table + "\n")

            buf.write("\n" + "=" * 75 + "\n")
            buf.write("  TORCH PROFILER - Top 15 ops by CPU time\n")
            buf.write("=" * 75 + "\n")
            table_cpu = all_key_averages[-1].table(
                sort_by="cpu_time_total", row_limit=15
            )
            buf.write(table_cpu + "\n")

            buf.write("\n" + "=" * 75 + "\n")
            buf.write("  TORCH PROFILER - Top 15 ops by GPU memory\n")
            buf.write("=" * 75 + "\n")
            table_mem = all_key_averages[-1].table(
                sort_by="self_cuda_memory_usage", row_limit=15
            )
            buf.write(table_mem + "\n")

            buf.write(f"\n  Chrome traces saved to: {trace_dir}/inference_*.json\n")
            buf.write(f"  Open in: chrome://tracing or https://ui.perfetto.dev\n\n")

        report = buf.getvalue()
        print(report)
        sys.stdout.flush()

        with open(f"{LOGDIR}/report.txt", "w") as f:
            f.write(report)

        if all_key_averages:
            with open(prof_table_path, "w") as f:
                f.write(table)
            print(f"[PROFILER] Table saved to {prof_table_path}")

        sys.stdout.flush()
        sys.stderr.flush()
        env.close()


if __name__ == "__main__":
    main()
