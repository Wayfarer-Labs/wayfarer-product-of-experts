from __future__ import annotations

from pathlib import Path

import base64
import html
import imageio.v2 as imageio
import numpy as np
from IPython.display import HTML, display


class HorizontalVisBuffer:
    """
    Collect multiple (name, frames) pairs and emit a single side-by-side GIF.

    frames_per_gif : expected number of frames per sequence
    fps            : playback FPS for the output GIF
    out_dir        : where to save the file; created if missing
    """

    def __init__(
        self,
        *,
        frames_per_gif: int,
        fps:            int = 24,
        out_dir:        str | Path = "visualizations",
    ):
        self.frames_per_gif = int(frames_per_gif)
        self.fps            = int(fps)
        self.out_dir        = Path(out_dir)
        self._buffer: list[tuple[str, list[np.ndarray]]] = []


    def push(self, name: str, frames: list[np.ndarray]) -> None:
        """
        Add a clip.

        * `frames` must be a list/array of HxWx3 uint8 numpy arrays
        * All pushed clips must have identical length H, W and dtype.
        """
        if len(frames) != self.frames_per_gif:
            raise ValueError(
                f"{name}: expected {self.frames_per_gif} frames, got {len(frames)}"
            )

        if not self._buffer:  # first clip – record shape/dtype
            self._h, self._w, self._c = frames[0].shape
            self._dtype = frames[0].dtype
        else:  # assert compatibility
            h, w, c = frames[0].shape
            if (h, w, c) != (self._h, self._w, self._c):
                raise ValueError(
                    f"{name}: frame shape {h,w,c} "
                    f"does not match previous {(self._h, self._w, self._c)}"
                )
            if frames[0].dtype != self._dtype:
                raise ValueError(
                    f"{name}: dtype {frames[0].dtype} "
                    f"does not match previous {self._dtype}"
                )

        self._buffer.append((name, frames))

    def display(self, *, filename: str | None = None, width_per_clip: int = 320) -> None:
        """
        Stack all pushed clips horizontally, render to GIF, and show inline.
        Clears nothing; call `clear()` when you're done.
        """
        if not self._buffer:
            print("HorizontalVisBuffer: nothing to display.")
            return

        # 1. stack corresponding frames side-by-side
        n_clips = len(self._buffer)
        stitched: list[np.ndarray] = []
        for t in range(self.frames_per_gif):
            row = np.concatenate(
                [frames[t] for _, frames in self._buffer], axis=1  # concat width-wise
            )
            stitched.append(row)

        # 2. write GIF
        self.out_dir.mkdir(parents=True, exist_ok=True)
        filename = filename or "vis_" + "_".join(name for name, _ in self._buffer) + ".gif"
        gif_path = self.out_dir / filename
        imageio.mimsave(gif_path, stitched, fps=self.fps)

        # 3. embed inline
        data_uri = (
            "data:image/gif;base64,"
            + base64.b64encode(gif_path.read_bytes()).decode()
        )

        captions = "&nbsp;&nbsp;|&nbsp;&nbsp;".join(
            html.escape(name) for name, _ in self._buffer
        )
        display(
            HTML(
                f"<div style='text-align:center'>"
                f"<div style='margin-bottom:4px'>{captions}</div>"
                f"<img src='{data_uri}' width='{width_per_clip * n_clips}'>"
                f"</div>"
            )
        )

    def clear(self) -> None: self._buffer.clear()

    def __len__(self) -> int: return len(self._buffer)
