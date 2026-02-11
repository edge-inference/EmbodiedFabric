"""Recording utilities for TDWBackend."""

from __future__ import annotations

import glob
import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class TDWRecordingMixin:
    def _setup_recording(self) -> None:
        try:
            from tdw.add_ons.third_person_camera import ThirdPersonCamera
            from tdw.add_ons.image_capture import ImageCapture

            def next_run_id(base_dir: str) -> int:
                try:
                    existing = [
                        int(name.split("_")[1])
                        for name in os.listdir(base_dir)
                        if name.startswith("run_") and name.split("_")[1].isdigit()
                    ]
                    return max(existing, default=0) + 1
                except FileNotFoundError:
                    return 1

            run_id = next_run_id(self._recording_path)
            self._recording_path = os.path.join(self._recording_path, f"run_{run_id}")
            os.makedirs(self._recording_path, exist_ok=True)

            width, height = getattr(self.config, "recording_resolution", (1280, 720))
            self._controller.communicate(
                [
                    {"$type": "set_screen_size", "width": width, "height": height},
                    {"$type": "set_render_quality", "render_quality": 2},
                    {"$type": "set_post_process", "value": False},
                    {"$type": "set_shadow_strength", "strength": 0.5},
                ]
            )

            scene_name = getattr(self.config, "scene_name", "") or ""
            if scene_name.startswith("floorplan"):
                camera_pos = {"x": -8, "y": 12, "z": -12}
                camera_look = {"x": 1, "y": 0, "z": 2}
                fov = 68
            else:
                camera_pos = {"x": -5, "y": 25, "z": -8}
                camera_look = {"x": 0, "y": 1.5, "z": 0}
                fov = 60

            self._third_person_camera = ThirdPersonCamera(
                position=camera_pos,
                look_at=camera_look,
                avatar_id="overhead_cam",
                field_of_view=fov,
            )

            self._image_capture = ImageCapture(
                avatar_ids=["overhead_cam"],
                path=self._recording_path,
                png=False,
            )

            self._controller.add_ons.extend([self._third_person_camera, self._image_capture])
            self._controller.communicate([])

            logger.info("Recording enabled: %s", self._recording_path)
        except Exception as e:
            logger.warning("Failed to setup recording: %s", e)
            self._recording_enabled = False

    def _compile_video(self, fps: int = 30) -> Optional[str]:
        video_path = os.path.join(self._recording_path, "simulation.mp4")
        frames_dir = os.path.join(self._recording_path, "overhead_cam")

        jpg_frames = glob.glob(os.path.join(frames_dir, "img_*.jpg"))
        png_frames = glob.glob(os.path.join(frames_dir, "img_*.png"))
        frames_pattern = (
            os.path.join(frames_dir, "img_%04d.jpg")
            if len(jpg_frames) > len(png_frames)
            else os.path.join(frames_dir, "img_%04d.png")
        )

        if not os.path.exists(frames_dir):
            logger.warning("No frames found in %s", frames_dir)
            return None

        total_frames = len(jpg_frames) + len(png_frames)
        if total_frames == 0:
            logger.warning("No image frames found in %s", frames_dir)
            return None

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                frames_pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                video_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Video saved: %s (%d frames)", video_path, self._frame_count)
                return video_path
            logger.warning("ffmpeg failed: %s", result.stderr)
            return None
        except FileNotFoundError:
            logger.warning("ffmpeg not found; frames kept at %s", frames_dir)
            return None

