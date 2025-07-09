from __future__ import annotations
import math, functools, contextlib
from tqdm.auto import tqdm
import numpy as np
import torch
from pathlib import Path
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



def _cube_grid(ax, lim=(0, 1), step=0.25, lw=0.3):
    ticks = np.arange(lim[0], lim[1] + 1e-9, step)
    for t in ticks:
        ax.plot([lim[0], lim[1]], [t, t], [lim[0], lim[0]], c="k", lw=lw, alpha=0.3)
        ax.plot([lim[0], lim[1]], [t, t], [lim[1], lim[1]], c="k", lw=lw, alpha=0.3)
        ax.plot([t, t], [lim[0], lim[1]], [lim[0], lim[0]], c="k", lw=lw, alpha=0.3)
        ax.plot([t, t], [lim[0], lim[1]], [lim[1], lim[1]], c="k", lw=lw, alpha=0.3)
        ax.plot([t, t], [lim[0], lim[0]], [lim[0], lim[1]], c="k", lw=lw, alpha=0.3)
        ax.plot([t, t], [lim[1], lim[1]], [lim[0], lim[1]], c="k", lw=lw, alpha=0.3)
        ax.plot([lim[0], lim[0]], [t, t], [lim[0], lim[1]], c="k", lw=lw, alpha=0.3)
        ax.plot([lim[1], lim[1]], [t, t], [lim[0], lim[1]], c="k", lw=lw, alpha=0.3)


def fwd_rectified_flow(points, t, noise=None):
    noise = torch.rand_like(points) if noise is None else noise
    return (1 - t) * points + t * noise


class _Renderer3D:

    def __init__(
        self,
        lim=(0, 1),
        elev=20,
        azim=45,
        dot_size=10,
        max_history=1000,
    ):
        self.lim = lim
        self.elev, self.azim, self.dot_size = elev, azim, dot_size

        # figure & axes
        self.fig = plt.figure(figsize=(4, 4), dpi=120)
        self.ax: Axes3D = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=elev, azim=azim)
        _ = self.ax.set_xlim(*lim), self.ax.set_ylim(*lim), self.ax.set_zlim(*lim)
        _ = self.ax.set_xticks([]), self.ax.set_yticks([]), self.ax.set_zticks([])
        with contextlib.suppress(AttributeError):
            self.ax.set_box_aspect([1, 1, 1])

        _cube_grid(self.ax, lim=lim)

        # place-holders, allocated lazily on first call
        self._scatter = None
        self._paths: list[matplotlib.lines.Line2D] = []
        self._txt = self.ax.text2D(0.02, 0.96, "", transform=self.ax.transAxes,
                                   fontsize=9, color="black",
                                   bbox=dict(facecolor="white", alpha=0.6, pad=1))
        self.max_history = max_history

    def render(
        self,
        pts: torch.Tensor,        # [N,6] on *any* device / dtype
        t: float,
        path_history: np.ndarray | None,    # [H,N,3]
    ) -> np.ndarray:
        pts = pts.detach().cpu().numpy()
        xyz, rgb = pts[:, :3], pts[:, 3:].clip(0, 1)

        if self._scatter is None:
            self._scatter = self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                                            s=self.dot_size, depthshade=True, c=rgb)
        else:
            self._scatter._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
            self._scatter.set_facecolors(rgb)

        # clean old line objects if N changed:
        if path_history is None:
            for ln in self._paths:
                ln.set_visible(False)
            self._paths.clear()
        else:
            H, N, _ = path_history.shape
            # allocate missing Line3D objects if N changed
            while len(self._paths) < N:
                ln, = self.ax.plot([], [], [], c="k", lw=0.3, alpha=0.1)
                self._paths.append(ln)
            for i, ln in enumerate(self._paths):
                if i < N:
                    ln.set_data_3d(path_history[:, i, 0],
                                   path_history[:, i, 1],
                                   path_history[:, i, 2])
                    ln.set_alpha(0.1)
                    ln.set_visible(True)
                else:
                    ln.set_visible(False)

        self._txt.set_text(f"t = {t:0.2f}")

        self.fig.canvas.draw()
        img = np.asarray(self.fig.canvas.buffer_rgba())
        return img.copy()


# lazily create a global renderer instance
@functools.lru_cache(maxsize=1)
def _get_renderer(**kw):
    return _Renderer3D(**kw)


def _scatter_frame(points, t, *, path_xyz=None, **kw) -> np.ndarray:
    """Internally mapped onto the global renderer."""
    ren = _get_renderer(**kw)
    return ren.render(points, t, path_xyz)


def render_points_over_time(traj: torch.Tensor, *,
                            path: str | Path | None = None,
                            show_path=False,
                            progress_bar: tqdm | None = None) -> list[np.ndarray] | None:
    traj = traj.unsqueeze(0) if traj.ndim == 2 else traj
    tvals = torch.linspace(0, 1, len(traj))

    history: list[np.ndarray] = []
    frames:  list[np.ndarray] = []
    for pts, t in zip(traj, tvals):
        if progress_bar: progress_bar.update(1)
        history.append(pts[:, :3].cpu().detach().numpy())
        path_xyz = np.stack(history[-_get_renderer().max_history :], 0) if show_path else None
        frames.append(_scatter_frame(pts, t, path_xyz=path_xyz))

    out_path = path and Path(path)
    if not out_path:            return frames
    elif len(frames) == 1:      imageio.imwrite(out_path.with_suffix(".png"), frames[0])
    else:                       imageio.mimsave(out_path.with_suffix(".gif"), frames, duration=0.06)


def visualize_noising_demo(points: torch.Tensor, *, T=30, out="noise_denoise", show_path=False):
    ts = torch.linspace(0, 1, T)
    eps = torch.rand_like(points)
    forward = torch.stack([fwd_rectified_flow(points, t, eps) for t in ts])
    reverse = torch.stack([fwd_rectified_flow(points, t, eps) for t in ts.flip(0)])
    traj = torch.cat([forward, reverse])
    render_points_over_time(traj, path=out, show_path=show_path)
