# SmolVLA SO101 Pick-Orange Evaluation Evidence

This record preserves the compact outcomes of the local LeIsaac evaluation
runs whose videos, per-step JSONL traces, and verbose logs were removed during
workspace cleanup. The task instruction was `pick up the orange and place it
on the plate`.

## Five-seed evaluation

Each seed used 20 episodes and the standard SO101 rest-pose tolerance.

| Checkpoint | Hugging Face revision | Seed 42 | Seed 123 | Seed 2026 | Seed 7 | Seed 1234 | Aggregate |
|---|---|---:|---:|---:|---:|---:|---:|
| v2, 100 episodes | `v2-100-ep` | 9/20 | 9/20 | 8/20 | 8/20 | 9/20 | 43/100 (43%) |
| combined, 220 episodes | `train/220-episodes` | 10/20 | 11/20 | 9/20 | 11/20 | 13/20 | 54/100 (54%) |

The 220-episode sweep ran from 2026-05-06 17:33 to 22:48 PDT. The v2
100-episode sweep ran from 2026-05-06 23:04 to 2026-05-07 04:47 PDT.

## Wrist-roll slack diagnostic

These diagnostic sweeps used 10 episodes per seed and widened the SO101
`wrist_roll` rest-pose acceptance interval from +/-30 degrees to +/-55 degrees.
They test termination sensitivity and must not be compared directly with the
standard-tolerance results as an independent policy improvement.

| Checkpoint | Seed 42 | Seed 123 | Seed 2026 | Aggregate |
|---|---:|---:|---:|---:|
| v2, 100 episodes | 6/10 | 3/10 | 9/10 | 18/30 (60.0%) |
| combined, 220 episodes | 7/10 | 8/10 | 8/10 | 23/30 (76.7%) |

The strict v2 diagnostic began with seed 42 and obtained 1/10 before the
remaining sweep was interrupted. It is incomplete and is not an aggregate
three-seed result.

## Published checkpoint provenance

All removed checkpoint weights were verified against their public Hugging Face
LFS SHA-256 values before deletion.

| Artifact | Published source | SHA-256 |
|---|---|---|
| SmolVLA v2 model | `edge-inference/smolvla-so101-pick-orange`, revision `v2-100-ep` | `d4f509ab1e873c15440cc68efd97fbe7b49145dfb9626ad30cb74d2de64c840b` |
| SmolVLA 220-episode model | `edge-inference/smolvla-so101-pick-orange`, revision `train/220-episodes` | `66d984a8800d2266d19d9c0b7f12457551601e2c95ef45c72cd86fa9733cf3ba` |
| GR00T shard 1 | `nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation`, revision `629479f` | `6df39ff02e5287c67699c3fb251994fca404805ae11b7f9b06b5a8d7fa36df92` |
| GR00T shard 2 | same | `0496aee159747f5ce42136b0a866c158c08df16d3aa43d6724b4944a35142b70` |
| GR00T optimizer | same | `da1d0194b05261038afb2b06835fa88f381babc6b4e770338bef1802922c0004` |
