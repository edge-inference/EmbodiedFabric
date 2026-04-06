# Extern Patches

Patches for upstream repos that are not tracked as submodules.
Apply after cloning the upstream repo at the pinned commit.

## IsaacLab-Arena

- **Upstream**: https://github.com/isaac-sim/IsaacLab-Arena.git
- **Branch**: `release/0.1.1`
- **Pinned commit**: `755e8cf393165bc947fb3fb4fb07aaaa0e5dded0`

### Setup

```bash
cd extern
git clone -b release/0.1.1 https://github.com/isaac-sim/IsaacLab-Arena.git
cd IsaacLab-Arena
git checkout 755e8cf393165bc947fb3fb4fb07aaaa0e5dded0
git apply ../../patches/isaaclab-arena-modifications.patch
git apply ../../patches/isaaclab-arena-n16-config.patch
```

### Patches

| File | Description |
|------|-------------|
| `isaaclab-arena-modifications.patch` | G1 policy runner enhancements, teleop improvements, WBC controller updates, background library, closedloop config |
| `isaaclab-arena-n16-config.patch` | GR00T N1.6 locomanipulation config |
