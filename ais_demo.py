from __future__ import annotations

"""Manim scene: 3‑step Annealed Importance Sampling demo (colour‑based).

Eight coloured particles illustrate a playful 3‑step *Annealed Importance
Sampling*–style procedure:

α = 0   → unit Gaussian (blue)
α = 0.5 → intermediate blend (orange)
α = 1   → sharp three‑mode PoE (red)

At each stage, dots perform a short "Langevin wiggle" driven by a finite‑
difference gradient of `log f_α` plus noise, then a *colour‑based* resampling:
keep the 40 % of dots whose RGB is closest to reference **purple** `#800080`,
fade the rest, and duplicate winners so the population size stays constant.

This is *not* statistically correct AIS—just a fun visual.
"""

from typing import Tuple, List
import random
import numpy as np
from manim import *
from manim.utils.color import ManimColor

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PARTICLE_RADIUS = 0.09
N_PARTICLES     = 16

TARGET_RGB_INT  = np.array([0, 128, 0])                 # purple reference
TARGET_COLOR    = ManimColor(tuple(TARGET_RGB_INT))
KEEP_RATIO      = 0.4                                     # survivors per step

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def rand_rgb_int() -> Tuple[int, int, int]:
    """Random 24‑bit colour as three ints (0‑255)."""
    return random.choice([(0, 0, 255), (0, 255, 0), (255, 0, 0)])


def color_int_to_float(rgb: Tuple[int, int, int]) -> np.ndarray:
    return np.array(rgb) / 255.0


def rgb_distance(rgb_a: np.ndarray, rgb_b: np.ndarray) -> float:
    return np.linalg.norm(rgb_a.astype(float) - rgb_b.astype(float))

TARGET_RGB_FLOAT = color_int_to_float(tuple(TARGET_RGB_INT))

# ─────────────────────────────────────────────────────────────────────────────
# Scene
# ─────────────────────────────────────────────────────────────────────────────

class AISDemo(Scene):
    current_alpha: float  # track current α so resample can project clones

    # ──────────────────── main animation ────────────────────
    def construct(self):
        self.axes = Axes(x_range=[-4, 4, 1],
                          y_range=[0, 1.1, 0.2],
                          tips=False)
        label = Text("Product-of-Experts for Visual Generation", font_size=24).to_corner(UP)
        self.play(Create(self.axes), FadeIn(label, shift=DOWN), run_time=0.3)

        # curves for α = 0, 0.5, 1
        curve_p4 = self.blended_curve(alpha=0.0, color=BLUE)
        curve_p3 = self.blended_curve(alpha=0.3, color=YELLOW)
        curve_p2 = self.blended_curve(alpha=0.65, color=ORANGE)
        curve_p1 = self.blended_curve(alpha=1.0, color=RED)

        # initial particles under α = 0
        self.current_alpha = 0.0
        particles = self.init_particles(N_PARTICLES)

        # stage p3 (α = 0)
        # showtext: we want to sample from a product-of-experts


        self.show_distribution(curve_p3, particles)

        # p3 → p2
        self.morph_curve(curve_p3, curve_p2)
        p2_particles = self.langevin_wiggle(particles, target_alpha=0.5)
        self.morph_particles(particles, p2_particles)
        self.current_alpha = 0.5
        p2_particles = self.resample(p2_particles, add_back=True)

        # p2 → p1
        particles = self.langevin_wiggle(particles, target_alpha=1.0)
        self.current_alpha = 1.0
        particles = self.resample(particles, add_back=False)
        self.morph_curve(curve_p2, curve_p1)

        self.wait(1)

    # ──────────────────── curve helpers ────────────────────
    def blended_curve(self, alpha: float, *, color: ManimColor):
        """Return VMobject for (1‑α)·N(0,1) + α·3‑peaked PoE."""
        graph = self.axes.plot(lambda x: self.blend_pdf_scalar(x, alpha),
                               color=color, stroke_width=4)
        return graph

    # pdf for scalar x and alpha
    def blend_pdf_scalar(self, x: float, alpha: float):
        g = np.exp(-0.5 * x**2)
        poe =  (0.5 * np.exp(-0.5 * (x + 2)**2 / 0.1) +
                0.2 * np.exp(-0.5 * x**2 / 0.1) +
                0.3 * np.exp(-0.5 * (x - 2)**2 / 0.1))
        return (1 - alpha) * g + alpha * poe

    # ──────────────────── particle helpers ────────────────────
    def init_particles(self, n: int) -> VGroup:
        """Place n particles sampled from N(0,1) along graph for current α."""
        xs = np.random.normal(size=n)
        dots = VGroup()
        for x in xs:
            y = self.blend_pdf_scalar(x, self.current_alpha)
            pos = self.axes.coords_to_point(x, y)
            rgb_int = rand_rgb_int()
            dot_color = ManimColor(rgb_int)
            dots.add(Dot(point=pos, radius=PARTICLE_RADIUS, color=dot_color))
        self.add(dots)
        return dots

    def show_distribution(self, graph: VMobject, particles: VGroup):
        self.play(Create(graph),
                  FadeIn(particles, scale=0.5),
                  run_time=1.5)
        self.wait(0.6)

    def morph_curve(self, curve_from: VMobject, curve_to: VMobject, with_trace: bool = True):
        if with_trace:
            self.add(curve_from.copy().set_stroke(GREY_B, width=1.5, opacity=0.6))
        self.play(ReplacementTransform(curve_from, curve_to), run_time=1.5)
        self.wait(0.3)

    def morph_particles(self, particles_from: VGroup, particles_to: VGroup):
        self.add(particles_from.copy().set_stroke(GREY_B, width=0.5, opacity=0.6))
        self.play(ReplacementTransform(particles_from, particles_to), run_time=1.5)
        self.wait(0.3)

    # ──────────────────── Langevin move ────────────────────
    def langevin_wiggle(self,
                        particles: VGroup,
                        *,
                        target_alpha: float,
                        steps: int = 8,
                        step_size: float = 0.05,
                        noise_scale: float = 0.3):
        label = Text("move ∝ ∂ log fα / ∂x", font_size=24).to_corner(UL)
        self.play(FadeIn(label, shift=DOWN), run_time=0.3)

        for _ in range(steps):
            for d in particles:
                # coordinates → data space
                x, _ = self.axes.point_to_coords(d.get_center())

                # finite‑difference grad of log‑pdf
                h = 1e-2
                f = lambda z: self.blend_pdf_scalar(z, target_alpha)
                grad = (np.log(f(x + h)) - np.log(f(x - h))) / (2 * h)

                dx = step_size * grad + noise_scale * np.random.randn()
                new_x = x + dx
                new_y = self.blend_pdf_scalar(new_x, target_alpha)
                d.move_to(self.axes.coords_to_point(new_x, new_y))
            self.wait(0.35)

        self.play(FadeOut(label, shift=DOWN), run_time=0.2)
        return particles

    # ──────────────────── colour‑based resample ────────────────────
    def resample(self, particles: VGroup, *, jitter_std: float = 0.2, add_back: bool = True) -> VGroup:
        """Fade non‑purple dots, duplicate winners, return new population."""
        # distance of each dot's *integer* RGB to target
        dists: List[float] = [
            rgb_distance(np.array(dot.get_color().to_rgb()) * 255, TARGET_RGB_INT)
            for dot in particles
        ]
        order = np.argsort(dists)
        k = max(1, int(len(particles) * KEEP_RATIO))
        keep_idx = order[:k]
        lose_idx = order[k:]

        # fade losers (slower so viewer sees it)
        self.play(*[FadeOut(particles[i], scale=0.2) for i in lose_idx],
                  run_time=1.0)
        self.wait(0.3)

        # collect winners
        winners = [particles[i] for i in keep_idx]
        new_particles = VGroup(*winners)  # start with the kept ones

        rng = np.random.default_rng()
        # duplicate until we restore original size
        while len(new_particles) < len(particles) and add_back:
            base = random.choice(winners)            # python's choice → no ndarray cast
            clone = base.copy()

            # small x‑jitter, then re‑project to curve at current α
            dx = jitter_std * rng.normal()
            x, _ = self.axes.point_to_coords(base.get_center())
            new_x = x + dx
            new_y = self.blend_pdf_scalar(new_x, self.current_alpha)
            clone.move_to(self.axes.coords_to_point(new_x, new_y))
            new_particles.add(clone)

        self.add(new_particles)
        return new_particles
