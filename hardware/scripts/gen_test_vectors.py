"""Extract test vectors from GR00T eval data for SystemC preprocessing testbench."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEOS_DIR = REPO_ROOT / "logs" / "videos"
OUT_DIR = Path(__file__).resolve().parent.parent / "tb" / "test_vectors"

STATE_MEAN = np.array([11.870278, -60.275116, 63.888458, 85.215569, -100.0, 8.776189], dtype=np.float32)
STATE_STD = np.array([28.037504, 42.821941, 32.673199, 15.687054, 1e-8, 12.104805], dtype=np.float32)

ACTION_MEAN = np.array([11.940205, -60.585850, 62.940823, 85.044769, 99.998734, 8.566985], dtype=np.float32)
ACTION_STD = np.array([28.240425, 42.548660, 33.463131, 16.068785, 0.085180, 12.482785], dtype=np.float32)

TARGET_SIZE = 512


def resize_with_pad(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    y_off = (target - new_h) // 2
    x_off = (target - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def img_to_float_chw(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)


def normalize_state(state: np.ndarray) -> np.ndarray:
    std = np.where(np.abs(STATE_STD) < 1e-8, 1e-8, STATE_STD)
    return (state - STATE_MEAN) / std


def unnormalize_action(action: np.ndarray) -> np.ndarray:
    return action * ACTION_STD + ACTION_MEAN


def save_bin(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(str(path))


def extract_frames(video_path: Path, frame_indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = {}
    idx = 0
    target_set = set(frame_indices)
    while cap.isOpened() and len(frames) < len(target_set):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in target_set:
            frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()
    return [frames[i] for i in frame_indices if i in frames]


def main():
    episode = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    robot_video = VIDEOS_DIR / f"episode_{episode}_robot.mp4"
    steps_file = VIDEOS_DIR / f"episode_{episode}_steps.jsonl"

    if not robot_video.exists() or not steps_file.exists():
        print(f"Missing eval data for episode {episode}")
        sys.exit(1)

    with open(steps_file) as f:
        steps = [json.loads(line) for line in f]

    total_steps = len(steps)
    sample_indices = np.linspace(0, total_steps - 1, n_samples, dtype=int).tolist()
    print(f"Extracting {n_samples} samples from episode {episode} ({total_steps} steps)")
    print(f"  Indices: {sample_indices}")

    # robot video is front+wrist side by side (1280x480)
    frames = extract_frames(robot_video, sample_indices)

    for i, (fi, frame) in enumerate(zip(sample_indices, frames)):
        step = steps[fi]
        out = OUT_DIR / f"sample_{i}"

        h, w = frame.shape[:2]
        mid = w // 2
        front_raw = frame[:, :mid]
        wrist_raw = frame[:, mid:]

        # raw inputs
        save_bin(front_raw.astype(np.uint8), out / "front_raw.bin")
        save_bin(wrist_raw.astype(np.uint8), out / "wrist_raw.bin")

        state = np.array(step["joint_pos"], dtype=np.float32)
        save_bin(state, out / "state_raw.bin")

        # preprocessed (expected output from the SoC)
        front_resized = resize_with_pad(front_raw, TARGET_SIZE)
        wrist_resized = resize_with_pad(wrist_raw, TARGET_SIZE)

        front_tensor = img_to_float_chw(front_resized)
        wrist_tensor = img_to_float_chw(wrist_resized)
        state_norm = normalize_state(state)

        save_bin(front_tensor, out / "expected" / "front_chw_f32.bin")
        save_bin(wrist_tensor, out / "expected" / "wrist_chw_f32.bin")
        save_bin(state_norm, out / "expected" / "state_norm_f32.bin")

        # metadata
        meta = {
            "step": fi,
            "front_shape": list(front_raw.shape),
            "wrist_shape": list(wrist_raw.shape),
            "state_raw": step["joint_pos"],
            "state_normalized": state_norm.tolist(),
            "target_size": TARGET_SIZE,
        }
        with open(out / "meta.json", "w") as mf:
            json.dump(meta, mf, indent=2)

        print(f"  Sample {i}: step={fi}, front={front_raw.shape}, wrist={wrist_raw.shape}")

    # save normalization constants for the SoC
    consts = OUT_DIR / "normalization"
    save_bin(STATE_MEAN, consts / "state_mean.bin")
    save_bin(STATE_STD, consts / "state_std.bin")
    save_bin(ACTION_MEAN, consts / "action_mean.bin")
    save_bin(ACTION_STD, consts / "action_std.bin")
    print(f"\nTest vectors saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
