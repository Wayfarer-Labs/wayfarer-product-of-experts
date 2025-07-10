from typing import Tuple, List, Optional
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


class AlgorithmStepDisplay(VGroup):
    """A stateful component that displays algorithm steps with highlighting using LaTeX."""
    
    def __init__(
        self,
        font_size: float = 20,
        line_spacing: float = 0.3,
        inactive_opacity: float = 0.6,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.font_size = font_size
        self.line_spacing = line_spacing
        self.inactive_opacity = inactive_opacity
        
        # Define the algorithm steps with their indentation levels
        # Using LaTeX notation with proper indentation
        self.steps = [
            (0, r"\text{Initialize } L \text{ particles} \sim \mathcal{N}(0, 1)", "0"),
            (0, r"\text{for } t = 0 \rightarrow T:", "1"),
            (1, r"\text{for } k = 0 \rightarrow K:", "2"),
            (2, r"\text{Langevin Sample}", "3"),
            (1, r"\text{resample based on scores}", "4"),
            (0, r"\text{choose winner from } L", "5")  # We'll handle coloring separately
        ]
        
        # State variables
        self.current_step = -1
        self.current_t = 0
        self.current_k = 0
        self.max_t = 3
        self.max_k = 3
        
        # Store indentation levels for positioning
        self.indent_levels = [indent for indent, _, _ in self.steps]
        
        # Create text objects for each step
        self.step_texts = []
        self.step_numbers = []
        
        for i, (indent, text, number) in enumerate(self.steps):
            # Replace placeholders in text
            display_text = text
            if i == 1:  # for t loop
                display_text = rf"\text{{for }} t = 0 \rightarrow {self.max_t}:"
            elif i == 2:  # for k loop
                display_text = rf"\text{{for }} k = 0 \rightarrow {self.max_k}:"
            
            # Don't add indentation to the text itself - we'll handle it through positioning
            
            # Create LaTeX text with explicit color
            if i == 5:  # Special handling for the last step with green "winner"
                # Split the text to color "winner" separately
                text_obj = VGroup(
                    MathTex(r"\text{choose }", font_size=self.font_size, color=WHITE),
                    MathTex(r"\text{winner}", font_size=self.font_size, color=GREEN),
                    MathTex(r"\text{ from } L", font_size=self.font_size, color=WHITE)
                ).arrange(RIGHT, buff=0.05)
            else:
                text_obj = MathTex(
                    display_text,
                    font_size=self.font_size,
                    color=WHITE
                )
            text_obj.set_z_index(10)  # Ensure visibility
            
            # Create step number in LaTeX
            number_obj = MathTex(
                number,
                font_size=self.font_size,
                color=WHITE
            )
            number_obj.set_z_index(10)  # Ensure visibility
            
            self.step_texts.append(text_obj)
            self.step_numbers.append(number_obj)
        
        # Arrange texts with proper indentation
        self._arrange_texts()
        
        # Store the original left positions for each text (to maintain after transforms)
        self.text_left_positions = []
        for text in self.step_texts:
            if isinstance(text, VGroup):
                self.text_left_positions.append(text[0].get_left()[0])
            else:
                self.text_left_positions.append(text.get_left()[0])
        
        # Initially hide all steps (they will fade in)
        for i, (text, number) in enumerate(zip(self.step_texts, self.step_numbers)):
            if isinstance(text, VGroup):  # Special handling for step 5
                for part in text:
                    part.set_opacity(0)
            else:
                text.set_opacity(0)
            number.set_opacity(0)
            self.add(text, number)
    
    def _arrange_texts(self):
        """Arrange text objects with proper spacing and alignment."""
        # Define starting positions and indentations
        base_x = 2.8  # Position for top-right
        base_y = 2.2  # Start below the title
        indent_width = 0.6  # Good visible indentation
        y_spacing = self.line_spacing  # Use the configured line spacing
        
        # Calculate positions for each step
        for i, text in enumerate(self.step_texts):
            # Calculate y position (going down from top)
            y_pos = base_y - (i * y_spacing)
            
            # Calculate x position based on indent level
            indent_level = self.indent_levels[i]
            x_pos = base_x + (indent_width * indent_level)
            
            # First, center the text at the target position
            text.move_to([x_pos, y_pos, 0])
            
            # Then adjust so the left edge is exactly at x_pos
            if isinstance(text, VGroup):
                # For VGroup, align based on the first element
                current_left = text[0].get_left()[0]
                text.shift(RIGHT * (x_pos - current_left))
            else:
                current_left = text.get_left()[0]
                text.shift(RIGHT * (x_pos - current_left))
        
        # Find the rightmost edge for number alignment
        max_right = -10
        for text in self.step_texts:
            right_x = text.get_right()[0]
            if right_x > max_right:
                max_right = right_x
        
        # Position numbers
        number_x = min(max_right + 0.7, 6.3)  # Ensure it fits on screen
        for text, number in zip(self.step_texts, self.step_numbers):
            # Position at same height as text
            number.move_to([number_x, text.get_center()[1], 0])
            # Align left edge of number
            number.shift([(number_x - number.get_left()[0]), 0, 0])
    
    def update_step(self, step: int, t: Optional[int] = None, k: Optional[int] = None):
        """Update the current step and optionally the loop variables."""
        self.current_step = step
        
        # Update visibility and colors - all steps remain visible
        for i, (text, number) in enumerate(zip(self.step_texts, self.step_numbers)):
            if i == step:
                # Current step - full white (or green for winner)
                if isinstance(text, VGroup):  # Special handling for step 5
                    text[0].set_color(WHITE).set_opacity(1)
                    text[1].set_color(GREEN).set_opacity(1)  # Keep winner green
                    text[2].set_color(WHITE).set_opacity(1)
                else:
                    text.set_color(WHITE).set_opacity(1)
                number.set_color(WHITE).set_opacity(1)
            else:
                # All other steps - grayed out but visible
                if isinstance(text, VGroup):  # Special handling for step 5
                    for part in text:
                        part.set_color(GRAY).set_opacity(self.inactive_opacity)
                else:
                    text.set_color(GRAY).set_opacity(self.inactive_opacity)
                number.set_color(GRAY).set_opacity(self.inactive_opacity)
    
    def animate_to_step(self, step: int, t: Optional[int] = None, k: Optional[int] = None):
        """Return animations to transition to a new step."""
        animations = []
        old_step = self.current_step
        
        # Handle text updates for loop variables first
        if t is not None and self.current_t != t and old_step >= 1:
            self.current_t = t
            old_text = self.step_texts[1]
            
            new_text = MathTex(
                rf"\text{{for }} t = {t} \rightarrow {self.max_t}:",
                font_size=self.font_size,
                color=self.step_texts[1].get_color()
            )
            # Position new text with same alignment as old
            new_text.move_to(old_text.get_center())
            # Preserve left alignment
            new_left = new_text.get_left()[0]
            desired_left = self.text_left_positions[1]
            new_text.shift([(desired_left - new_left), 0, 0])
            
            animations.append(Transform(old_text, new_text))
            
        if k is not None and self.current_k != k and old_step >= 2:
            self.current_k = k
            old_text = self.step_texts[2]
            
            # Don't include indentation in the text - positioning handles it
            new_text = MathTex(
                rf"\text{{for }} k = {k} \rightarrow {self.max_k}:",
                font_size=self.font_size,
                color=self.step_texts[2].get_color()
            )
            # Position new text with same alignment as old
            new_text.move_to(old_text.get_center())
            # Preserve left alignment
            new_left = new_text.get_left()[0]
            desired_left = self.text_left_positions[2]
            new_text.shift([(desired_left - new_left), 0, 0])
            
            animations.append(Transform(old_text, new_text))
        
        # Update to new state
        self.update_step(step, t, k)
        
        # Handle visibility and color changes
        for i in range(len(self.step_texts)):
            if i <= step:
                if i > old_step:
                    # Newly visible step - fade in without shifting
                    animations.extend([
                        FadeIn(self.step_texts[i]),
                        FadeIn(self.step_numbers[i])
                    ])
                elif i == step and old_step != step:
                    # Newly current step - make white (or keep green for winner)
                    if isinstance(self.step_texts[i], VGroup):  # Special handling for step 5
                        animations.extend([
                            self.step_texts[i][0].animate.set_color(WHITE).set_opacity(1),
                            self.step_texts[i][1].animate.set_color(GREEN).set_opacity(1),
                            self.step_texts[i][2].animate.set_color(WHITE).set_opacity(1),
                            self.step_numbers[i].animate.set_color(WHITE).set_opacity(1)
                        ])
                    else:
                        animations.extend([
                            self.step_texts[i].animate.set_color(WHITE).set_opacity(1),
                            self.step_numbers[i].animate.set_color(WHITE).set_opacity(1)
                        ])
                elif i == old_step and i < step:
                    # Previously current step - gray out
                    if isinstance(self.step_texts[i], VGroup):  # Special handling for step 5
                        for part in self.step_texts[i]:
                            animations.append(part.animate.set_color(GRAY).set_opacity(self.inactive_opacity))
                    else:
                        animations.append(self.step_texts[i].animate.set_color(GRAY).set_opacity(self.inactive_opacity))
                    animations.append(self.step_numbers[i].animate.set_color(GRAY).set_opacity(self.inactive_opacity))
        return animations
    
    def fade_in_all(self):
        """Return animations to fade in all steps as grayed out."""
        animations = []
        for i, (text, number) in enumerate(zip(self.step_texts, self.step_numbers)):
            if isinstance(text, VGroup):  # Special handling for step 5
                for part in text:
                    animations.append(part.animate.set_opacity(self.inactive_opacity).set_color(GRAY))
            else:
                animations.append(text.animate.set_opacity(self.inactive_opacity).set_color(GRAY))
            animations.append(number.animate.set_opacity(self.inactive_opacity).set_color(GRAY))
        return AnimationGroup(*animations)


class AISDemo2(Scene):
    current_alpha: float

    def construct(self):
        # Set up axes - centered and larger
        self.axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.5, 0.1],  # Adjusted for sharp peaks
            tips=False,
            x_length=7,  # Larger for better visibility
            y_length=3.5,  # Proportional height
            axis_config={
                "include_numbers": True,
                "font_size": 20,
            }
        ).move_to(LEFT * 1.2 + DOWN * 0.2)  # Center-left and slightly down
        
        label = Text("Product-of-Experts for Visual Generation", font_size=28).to_edge(UP, buff=0.3).to_edge(LEFT, buff=0.5)
        
        label = Text("Product-of-Experts for Visual Generation", font_size=26).to_corner(UL, buff=0.5)  # Top-left
        
        # Create algorithm display with adjusted parameters
        self.algo_display = AlgorithmStepDisplay(
            font_size=24,  # Good size for visibility
            line_spacing=0.3,  # Tighter spacing
            inactive_opacity=0.6  # More visible when grayed out
        )
        self.algo_display.set_z_index(10)
        
        # Initial setup
        self.play(
            Create(self.axes), 
            FadeIn(label, shift=DOWN),
            self.algo_display.fade_in_all(),  # Fade in all algorithm steps as gray
            run_time=0.3
        )
        
        # curves for α = 0, 0.5, 1
        curve_p4 = self.blended_curve(alpha=0.0, color=BLUE)
        curve_p3 = self.blended_curve(alpha=0.3, color=YELLOW)
        curve_p2 = self.blended_curve(alpha=0.65, color=ORANGE)
        curve_p1 = self.blended_curve(alpha=1.0, color=RED)
        
        # Create initial curve
        self.play(Create(curve_p4), run_time=0.3)
        self.wait(0.2)
        
        # Step 0: Initialize particles
        self.play(self.algo_display.animate_to_step(0), run_time=0.3)
        self.wait(0.1)
        
        # Step 1: Start t loop
        self.play(self.algo_display.animate_to_step(1, t=0), run_time=0.2)
        self.wait(0.1)
        
        # Step 2: Start k loop
        self.play(self.algo_display.animate_to_step(2, t=0, k=0), run_time=0.2)
        self.wait(0.05)
        
        # Step 3: Langevin sample
        self.play(self.algo_display.animate_to_step(3, t=0, k=0), run_time=0.2)
        self.wait(0.05)
        
        # Continue k loop (very fast)
        for k in range(1, 4):
            self.play(self.algo_display.animate_to_step(2, t=0, k=k), run_time=0.1)
            self.play(self.algo_display.animate_to_step(3, t=0, k=k), run_time=0.1)
        
        # Step 4: Resample - synchronized with curve transition
        self.play(
            self.algo_display.animate_to_step(4, t=0),
            Transform(curve_p4, curve_p3),
            run_time=0.5  # Same timing for both
        )
        self.wait(0.1)
        
        # Continue with more t iterations (synchronized transitions)
        for t in range(1, 4):
            self.play(self.algo_display.animate_to_step(1, t=t), run_time=0.2)
            self.wait(0.1)
            
            for k in range(4):
                self.play(self.algo_display.animate_to_step(2, t=t, k=k), run_time=0.1)
                self.play(self.algo_display.animate_to_step(3, t=t, k=k), run_time=0.1)
            
            # Resample and transform curve together
            if t == 1:
                next_curve = curve_p2
            elif t == 2:
                next_curve = curve_p1
            else:
                next_curve = None
            
            if next_curve:
                self.play(
                    self.algo_display.animate_to_step(4, t=t),
                    Transform(curve_p4, next_curve),
                    run_time=0.5  # Same timing for both
                )
            else:
                self.play(self.algo_display.animate_to_step(4, t=t), run_time=0.5)
        
        # Step 5: Choose winner
        self.play(self.algo_display.animate_to_step(5), run_time=0.3)
        self.wait(0.5)
        
        # Final wait to see the complete state
        self.wait(1)

    def blended_curve(self, alpha: float, *, color: ManimColor):
        """Return VMobject for (1‑α)·N(0,1) + α·3‑peaked PoE."""
        graph = self.axes.plot(lambda x: self.blend_pdf_scalar(x, alpha),
                               color=color, stroke_width=8)  # Even thicker lines
        return graph

    # pdf for scalar x and alpha
    def blend_pdf_scalar(self, x: float, alpha: float):
        # Standard normal
        g = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        
        # Product of Experts (3 peaked distribution) with very sharp peaks
        # Reduced variance (0.03) for sharp peaks that showcase hostile landscape
        # Heights: left=0.8 (tallest), middle=0.3 (shortest), right=0.5 (middle)
        poe = (0.8 * np.exp(-0.8 * (x + 2)**2 / 0.06) + 
               0.3 * np.exp(-0.5 * x**2 / 0.12) + 
               0.5 * np.exp(-0.5 * (x - 2)**2 / 0.06))
        
        # Normalize poe to have similar scale as g
        poe = poe * 0.07
        
        return (1 - alpha) * g + alpha * poe