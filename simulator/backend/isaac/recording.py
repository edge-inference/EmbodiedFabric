"""Recording helpers for IsaacSimBackend."""

from __future__ import annotations

import glob
import logging
import os
import subprocess

import numpy as np

logger = logging.getLogger(__name__)


class IsaacRecordingMixin:
    def _setup_recording(self) -> None:
        try:
            from omni.isaac.sensor import Camera

            os.makedirs(self._recording_path, exist_ok=True)

            self._camera = Camera(
                prim_path="/World/RecordingCamera",
                position=np.array([-4.0, 18.0, -4.0]),
                frequency=30,
                resolution=(1280, 720),
                orientation=None,
            )
            self._world.scene.add(self._camera)
            logger.info("Recording enabled: %s", self._recording_path)
        except Exception as e:
            logger.warning("Failed to setup recording: %s", e)
            self._recording_enabled = False

    def _capture_frame(self) -> None:
        try:
            from PIL import Image

            rgba = self._camera.get_rgba()
            if rgba is None:
                return
            img = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))
            frame_path = os.path.join(self._recording_path, f"frame_{self._frame_count:06d}.jpg")
            img.save(frame_path, quality=90)
        except Exception:
            return

    def _compile_video(self) -> None:
        frames = glob.glob(os.path.join(self._recording_path, "frame_*.jpg"))
        if not frames:
            return

        video_path = os.path.join(self._recording_path, "simulation.mp4")
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                "30",
                "-pattern_type",
                "glob",
                "-i",
                os.path.join(self._recording_path, "frame_*.jpg"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
            subprocess.run(cmd, capture_output=True)
            logger.info("Video saved: %s", video_path)
        except Exception as e:
            logger.warning("Failed to compile video: %s", e)

