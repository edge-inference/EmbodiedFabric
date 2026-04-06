# Examples

Small runnable scripts. Most real robot eval entrypoints live in `scripts/eval/`.

## Servers

- `lerobot_server.py`: serves LeRobot policies (SmolVLA / Pi0) over HTTP (needs LeRobot deps).

## Simulator / backend demos

- `isaac_humanoid_test.py`: quick Isaac Sim spawn/step sanity check (needs Isaac Sim).

## Contracts

- `contracts.py`: assume/guarantee contract demos. Parameters come from `config/contracts.json`.
