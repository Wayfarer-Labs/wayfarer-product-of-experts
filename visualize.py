"""
Fast matplotlib-only version of the rectified-flow visualiser.
Same public API, ~20-40× faster for typical settings.
"""
from __future__ import annotations
import math, functools, contextlib
import numpy as np
import torch
from pathlib import Path
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")        # headless is fine – we grab pixels ourselves
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – side-effect import
import matplotlib.pyplot as plt

# ---------------------------- helpers ---------------------------------------


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


def sample_random_cloud(n, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.rand(n, 6)


def fwd_rectified_flow(points, t, noise=None):
    noise = torch.rand_like(points) if noise is None else noise
    return (1 - t) * points + t * noise


def _reflected_tvals(n):
    mid = (n - 1) // 2
    return [i / mid if i <= mid and mid > 0 else (n - 1 - i) / mid if mid > 0 else 0.0 for i in range(n)]


# ----------------------- singleton state for speed --------------------------
# Everything below lives once per interpreter.  All calls share the same
# figure/axes and simply update artists.

class _Renderer3D:
    """A minimal, re-usable 3-D scatter renderer."""

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

        # --- figure & axes ----------------------------------------------------
        self.fig = plt.figure(figsize=(4, 4), dpi=120)
        self.ax: Axes3D = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_xlim(*lim), self.ax.set_ylim(*lim), self.ax.set_zlim(*lim)
        self.ax.set_xticks([]), self.ax.set_yticks([]), self.ax.set_zticks([])
        with contextlib.suppress(AttributeError):
            self.ax.set_box_aspect([1, 1, 1])

        _cube_grid(self.ax, lim=lim)  # static; draw once

        # place-holders, allocated lazily on first call
        self._scatter = None
        self._paths: list[matplotlib.lines.Line2D] = []
        self._txt = self.ax.text2D(0.02, 0.96, "", transform=self.ax.transAxes,
                                   fontsize=9, color="black",
                                   bbox=dict(facecolor="white", alpha=0.6, pad=1))
        self.max_history = max_history

    # --------------------------------------------------------------------- #

    def render(
        self,
        pts: torch.Tensor,        # [N,6] on *any* device / dtype
        t: float,
        path_history: np.ndarray | None,    # [H,N,3]
    ) -> np.ndarray:
        pts = pts.detach().cpu().numpy()
        xyz, rgb = pts[:, :3], pts[:, 3:].clip(0, 1)

        # ---------------- scatter ----------------
        if self._scatter is None:
            self._scatter = self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                                            s=self.dot_size, depthshade=True, c=rgb)
        else:
            # _offsets3d is undocumented but fast
            self._scatter._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
            self._scatter.set_facecolors(rgb)

        # ---------------- history paths ----------
        # Clean old line objects if N changed:
        if path_history is None:
            for ln in self._paths:
                ln.set_visible(False)
            self._paths.clear()
        else:
            H, N, _ = path_history.shape
            # allocate missing Line3D objects
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

        # ---------------- text -------------------
        self._txt.set_text(f"t = {t:0.2f}")

        # ---------------- raster → np ------------ (fast – no PNG encoding)
        self.fig.canvas.draw()             # draw once
        img = np.asarray(self.fig.canvas.buffer_rgba())  # (h,w,4) uint8 view
        return img.copy()                  # detach from backend buffer


# Lazily create a global renderer instance
@functools.lru_cache(maxsize=1)
def _get_renderer(**kw):
    return _Renderer3D(**kw)


# -----------------------------------------------------------------------------
#                      public helpers (same signatures)
# -----------------------------------------------------------------------------


def _scatter_frame(points, t, *, path_xyz=None, **kw):
    """Internally mapped onto the global renderer."""
    ren = _get_renderer(**kw)
    return ren.render(points, t, path_xyz)


def render_points_over_time(traj: torch.Tensor, *, path: str | Path = "out", show_path=False):
    traj = traj.unsqueeze(0) if traj.ndim == 2 else traj
    tvals = _reflected_tvals(len(traj))

    history = []
    frames = []
    for pts, t in zip(traj, tvals):
        history.append(pts[:, :3].cpu().detach().numpy())
        path_xyz = np.stack(history[-_get_renderer().max_history :], 0) if show_path else None
        frames.append(_scatter_frame(pts, t, path_xyz=path_xyz))

    out_path = Path(path)
    if len(frames) == 1:
        imageio.imwrite(out_path.with_suffix(".png"), frames[0])
    else:
        imageio.mimsave(out_path.with_suffix(".gif"), frames, duration=0.06)


def visualize_noising_demo(points: torch.Tensor, *, T=30, out="noise_denoise", show_path=False):
    ts = torch.linspace(0, 1, T)
    eps = torch.rand_like(points)
    forward = torch.stack([fwd_rectified_flow(points, t, eps) for t in ts])
    reverse = torch.stack([fwd_rectified_flow(points, t, eps) for t in ts.flip(0)])
    traj = torch.cat([forward, reverse])
    render_points_over_time(traj, path=out, show_path=show_path)


# quick test ------------------------------------------------------------------
if __name__ == "__main__":
    from dataset import SPHERE_POINTS
    visualize_noising_demo(SPHERE_POINTS, T=40,
                           out="visualizations/sphere_noise_denoise_fast",
                           show_path=True)
