"""
One‑file demo: 1‑D Annealed Importance Sampling visualised with Manim
====================================================================
Changes in this version
-----------------------
* **Weight → colour mapping** rather than dot size.
* Particles start with **random hues**; as β progresses they fade
  toward *green* if highly‑weighted or *red* if ignored.

Quick usage
-----------
```bash
# 1. Run the sampler and save snapshots (CPU, ~1 s)
python ais_manim_vis.py --sample

# 2. Render at 480 p / 15 fps
manim ais_manim_vis.py AISVisualizer1D -q l -o ais_colour_demo.mp4
```
"""

import math, random, argparse, os
from pathlib import Path
from typing import Callable, List, Dict

import torch
from torch import Tensor

from manim import (
    Scene, Dot, Axes, ValueTracker, always_redraw, MathTex,
    interpolate_color, RED, GREEN, WHITE, UR, ManimColor, DOWN, LEFT, RIGHT, UP, OUT, IN
)

# -----------------------------------------------------------------------------
#                         Analytic 1‑D toy experts
# -----------------------------------------------------------------------------
class AnalyticGenerative1D:
    def velocity(self, x: Tensor, t: float) -> Tensor: ...

class AnalyticDiscriminative1D:
    def log_prob(self, x: Tensor) -> Tensor: ...

class GaussianFlow(AnalyticGenerative1D):
    def __init__(self, mu: float, sigma: float):
        self.mu, self.sigma = mu, sigma
    def velocity(self, x, t):
        return -(x - self.mu) / self.sigma ** 2

class GaussianLogPDF(AnalyticDiscriminative1D):
    def __init__(self, mu: float, sigma: float):
        self.mu, self.sigma = mu, sigma
        self._log_norm = -0.5 * math.log(2 * math.pi * sigma ** 2)
    def log_prob(self, x):
        return self._log_norm - 0.5 * ((x - self.mu) / self.sigma) ** 2

# -----------------------------------------------------------------------------
#                              AIS core sampler
# -----------------------------------------------------------------------------
@torch.no_grad()
def annealed_importance_sampling(
    generative_experts:     List[AnalyticGenerative1D],
    discriminative_experts: List[AnalyticDiscriminative1D],
    *,
    n_particles:        int   = 100,
    n_points:           int   = 512,
    n_denoise_steps:    int   = 128,
    n_seek_steps:       int   = 3,
    step_size:          float = 1.0,
    kappa:              float = 0.05,
    step_callback:      Callable[[Dict], None] | None = None,
) -> None:
    device              = torch.device("cpu")
    log_w               = torch.zeros(n_particles, device=device)
    ts                  = torch.linspace(1.0, 0.0, n_denoise_steps + 1, device=device)
    x                   = torch.randn(n_particles, n_points, device=device)  # noise ~ N(0,1)

    for i in range(n_denoise_steps):
        t_hi, t_lo = ts[i].item(), ts[i + 1].item()
        dt         = (t_lo - t_hi) / n_seek_steps

        # one Euler step using all generative experts
        velocities = [g.velocity(x, t_hi) for g in generative_experts]
        x += step_size * dt * sum(velocities)

        # Langevin refinement
        for _ in range(n_seek_steps):
            velocities = [g.velocity(x, t_hi) for g in generative_experts]
            scores     = sum(velocities)  # simple toy; normally convert v→score
            x += (kappa ** 2) / 2 * scores + kappa * torch.randn_like(x)

        # reward from discriminative experts
        log_w += sum(d.log_prob(x) for d in discriminative_experts).mean(dim=1)

        if step_callback is not None:
            step_callback({
                "step": i,
                "beta": t_hi,
                "x":    x[:, 0].clone().cpu(),  # take first dim for 1‑D plotting
                "logw": log_w.clone().cpu(),
            })

# -----------------------------------------------------------------------------
#                  Utility: run sampler & cache snapshots
# -----------------------------------------------------------------------------
SNAPSHOT_FILE = Path("ais_1d_snapshots.pt")

def run_sampler_and_collect() -> List[Dict]:
    snaps: List[Dict] = []

    gen  = [GaussianFlow(0.0, 4.0)]
    disc = [GaussianLogPDF(-2.0, 0.6), GaussianLogPDF(3.0, 1.0)]

    annealed_importance_sampling(
        gen, disc,
        n_particles=100,
        step_callback=lambda d: snaps.append(d),
    )
    torch.save(snaps, SNAPSHOT_FILE)
    return snaps

# -----------------------------------------------------------------------------
#                    Colour mapping helper (weight → hue)
# -----------------------------------------------------------------------------

def weight_to_colour(w: float) -> ManimColor:
    """0 ⇒ red, 1 ⇒ green."""
    return interpolate_color(RED, GREEN, w)

BASE_RADIUS = 0.08  # fixed dot size

# -----------------------------------------------------------------------------
#                               Manim scene
# -----------------------------------------------------------------------------
class AISVisualizer1D(Scene):
    def construct(self):
        # 1. Load or create snapshots
        if SNAPSHOT_FILE.exists():
            snaps = torch.load(SNAPSHOT_FILE)
        else:
            snaps = run_sampler_and_collect()

        # 2. Axes & PDF backbone (for visuals only)
        self.axes = Axes(
            x_range=[-6, 6, 1],
            y_range=[0, 0.35, 0.1],
            x_length=10,
            y_length=4,
            axis_config={"include_tip": False, "stroke_color": WHITE},
        ).to_edge(DOWN)
        self.add(self.axes)

        # 3. Particle dots with random initial hues
        self.dots = [
            Dot(color=ManimColor((random.random(), random.random(), random.random()))).set_width(BASE_RADIUS)
            for _ in range(len(snaps[0]["x"]))
        ]
        self.add(*self.dots)

        # 4. Beta label
        beta_tracker = ValueTracker(snaps[0]["beta"])
        beta_tex = always_redraw(
            lambda: MathTex(rf"\beta = {beta_tracker.get_value():.2f}")
            .scale(0.8)
            .to_corner(UR)
        )
        self.add(beta_tex)

        # 5. Density curve (target mixture) as background
        def target_pdf(x):
            return 0.3 * torch.exp(-0.5 * ((x + 2) / 0.6) ** 2) / (0.6 * math.sqrt(2 * math.pi)) + \
                   0.7 * torch.exp(-0.5 * ((x - 3) / 1.0) ** 2) / (1.0 * math.sqrt(2 * math.pi))
        density_curve = self.axes.plot(
            lambda x: target_pdf(torch.tensor(x)).item(),
            color=WHITE,
        )
        self.add(density_curve)

        # 6. Animation loop
        for snap in snaps:
            self._update_frame(snap, beta_tracker)
            self.wait(1 / 15)  # 15 fps for low‑quality render

    def _update_frame(self, snap: Dict, beta_tracker: ValueTracker):
        # Update β label
        beta_tracker.set_value(snap["beta"])

        # Weights → colours
        weights = torch.softmax(snap["logw"], dim=0)
        for dot, x_val, w in zip(self.dots, snap["x"], weights):
            dot.move_to(self.axes.c2p(float(x_val), 0))
            dot.set_width(BASE_RADIUS)
            dot.set_color(weight_to_colour(float(w)))

# -----------------------------------------------------------------------------
#                              CLI entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    pass