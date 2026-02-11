# Examples

Small runnable scripts. Most real robot eval entrypoints live in `scripts/eval/`.

## Servers

- `cogact_server.py`: serves CogACT policy over HTTP (needs CogACT deps).
- `lerobot_server.py`: serves LeRobot policies (SmolVLA / Pi0 / GR00T) over HTTP (needs LeRobot deps).
- `vlm_planner_server.py`: serves a VLM planner endpoint for `simulator.vla.vlm_planner`.
- `tdw_server.py`: starts a TDW build and prints its address (needs `tdw`).

## Simulator / backend demos

- `floorplan_4zone.py`: multi-robot coordination demo (TDW and/or Isaac backend, depending on config).
- `isaac_humanoid_test.py`: quick Isaac Sim spawn/step sanity check (needs Isaac Sim).

## Contracts

- `contracts.py`: assume/guarantee contract demos. Parameters come from `config/contracts.json`.

