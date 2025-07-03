import math, torch, numpy as np, matplotlib.pyplot as plt
import imageio.v2 as imageio
from io import BytesIO
from pathlib import Path


# -- helpers --
def _cube_grid(ax, lim: tuple = (0, 1), step: float = 0.25, lw: float = 0.3) -> None:
    """Draw faint gridlines on the bounding cube."""
    ticks = np.arange(lim[0], lim[1] + 1e-9, step)
    for t in ticks:
        # vertical faces
        ax.plot([lim[0], lim[1]], [t, t], [lim[0], lim[0]], c='k', lw=lw, alpha=0.3)
        ax.plot([lim[0], lim[1]], [t, t], [lim[1], lim[1]], c='k', lw=lw, alpha=0.3)
        ax.plot([t, t], [lim[0], lim[1]], [lim[0], lim[0]], c='k', lw=lw, alpha=0.3)
        ax.plot([t, t], [lim[0], lim[1]], [lim[1], lim[1]], c='k', lw=lw, alpha=0.3)
        # vertical lines
        ax.plot([t, t], [lim[0], lim[0]], [lim[0], lim[1]], c='k', lw=lw, alpha=0.3)
        ax.plot([t, t], [lim[1], lim[1]], [lim[0], lim[1]], c='k', lw=lw, alpha=0.3)
        ax.plot([lim[0], lim[0]], [t, t], [lim[0], lim[1]], c='k', lw=lw, alpha=0.3)
        ax.plot([lim[1], lim[1]], [t, t], [lim[0], lim[1]], c='k', lw=lw, alpha=0.3)


# -- sphere generator --
def golden_sphere_points(n: int,
                         seed: int | None = None) -> torch.Tensor:
    """
    Uniform-ish points on the unit sphere (centre 0.5).  
    Colours are random but repeatable via `seed`.
    Returns: [n,6] tensor  (xyz rgb)   all in [0,1]
    """
    if seed is not None:
        torch.manual_seed(seed)

    idx     = torch.arange(0, n, dtype=torch.float32) + 0.5
    phi     = torch.acos(1 - 2*idx/n)
    theta   = math.tau * idx * (1 + math.sqrt(5))/2

    x       = torch.cos(theta) * torch.sin(phi)
    y       = torch.sin(theta) * torch.sin(phi)
    z       = torch.cos(phi)

    xyz     = torch.stack([x, y, z], dim=1) * 0.5 + 0.5   # map to [0,1]
    rgb     = torch.rand(n, 3)                            # random colours [0,1]

    return torch.cat([xyz, rgb], dim=1)                   # [n,6]


SPHERE_POINTS = golden_sphere_points(512, seed=42)


# -- sampling utils --
def sample_random_cloud(n: int,
                        seed: int | None = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    return torch.rand(n, 6)


# -- rectified flow --
def fwd_rectified_flow(points: torch.Tensor,
                       t: float,
                       noise: torch.Tensor | None = None) -> torch.Tensor:
    noise = torch.rand_like(points) if noise is None else noise
    return (1 - t) * points + t * noise


def _reflected_tvals(n_frames: int) -> list[float]:
    """
    Produce [0,..,1,..,0] with the peak at the middle frame.
    Works for even or odd n_frames.
    """
    mid  = (n_frames - 1) // 2          # integer middle index
    vals = []
    for i in range(n_frames):
        if i <= mid:
            vals.append(i / mid if mid > 0 else 0.0)
        else:
            vals.append((n_frames - 1 - i) / mid if mid > 0 else 0.0)
    return vals


# -- updated scatter --
def _scatter_frame(points: torch.Tensor,
                   t: float,
                   lim: tuple = (0, 1),
                   elev: int = 20,
                   azim: int = 45,
                   dot_size: int = 10) -> np.ndarray:

    fig = plt.figure(figsize=(4, 4))
    ax  = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    xyz, rgb = points[:, :3], points[:, 3:]
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               c=rgb.clamp(0, 1).numpy(),
               s=dot_size, depthshade=True)

    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_zlim(*lim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    try: ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        rng = lim[1] - lim[0]
        for axis in 'xyz':
            getattr(ax, f'set_{axis}lim')(lim[0], lim[0] + rng)

    _cube_grid(ax, lim=lim)

    ax.text2D(0.02, 0.96, f"t = {t:0.2f}", transform=ax.transAxes,
              fontsize=9, color='black',
              bbox=dict(facecolor='white', alpha=0.6, pad=1))

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close(fig); buf.seek(0)
    return imageio.imread(buf)


# -- renderer with peak-at-mid --
def render_points_over_time(traj: torch.Tensor,
                            path: str | Path = "out") -> None:
    """
    traj : [T,N,6] or [N,6]
    Mid-frame gets t=1; ends get t=0.
    """
    traj  = traj.unsqueeze(0) if traj.ndim == 2 else traj
    tvals = _reflected_tvals(traj.shape[0])

    frames = [_scatter_frame(p, t) for p, t in zip(traj, tvals)]
    path   = Path(path)
    if len(frames) == 1:
        imageio.imwrite(path.with_suffix(".png"), frames[0])
    else:
        imageio.mimsave(path.with_suffix(".gif"), frames, duration=0.06)


# -- noise-denoise demo 
def visualize_noising_demo(points: torch.Tensor,
                      T: int = 30,
                      out: str | Path = "noise_denoise") -> None:
    ts      = torch.linspace(0, 1, T)
    eps     = torch.rand_like(points)
    forward = torch.stack([fwd_rectified_flow(points, t, eps) for t in ts])
    reverse = torch.stack([fwd_rectified_flow(points, t, eps) for t in ts.flip(0)])
    traj    = torch.cat([forward, reverse])          # [2T,N,6]
    render_points_over_time(traj, out)


# -- quick test --
if __name__ == "__main__":
    visualize_noising_demo(SPHERE_POINTS, T=40, out="sphere_noise_denoise")
