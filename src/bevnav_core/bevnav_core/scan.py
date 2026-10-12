"""
Six-heading pan scan sequencing (CLAUDE.md Stage 5), independent of ROS.

The pan joint is continuous and reports unwrapped angles, and its controller takes the long way
round when a wrapped target crosses +-pi. Targets are therefore unwrapped: base + k * step for
k = 0..headings-1, then base + headings * step (one full turn: facing forward again), where base
is the joint angle at the start rounded to whole turns. Every move is one short step.
"""

import math
from typing import Optional


class PanScan:
    """Decide pan targets and which frames to take; angles in rad, times in s (sim)."""

    def __init__(self, headings: int, step_rad: float, settle_tol_rad: float,
                 settle_time_s: float, timeout_s: float) -> None:
        self.headings, self.step = headings, step_rad
        self.tol, self.settle_s, self.timeout_s = settle_tol_rad, settle_time_s, timeout_s
        self.active = False

    def start(self, now_s: float, joint_rad: float) -> float:
        """Begin a scan; return the first pan target."""
        self.base = 2 * math.pi * round(joint_rad / (2 * math.pi))
        self.k, self.t0, self.settled_since = 0, now_s, None
        self.active = True
        return self.target()

    def target(self) -> float:
        """Return the unwrapped pan goal; after the last view, the full-turn home angle."""
        return self.base + self.k * self.step

    def on_joint(self, now_s: float, joint_rad: float) -> None:
        """Track when the joint reached the target and stayed within the tolerance."""
        if abs(joint_rad - self.target()) < self.tol:
            if self.settled_since is None:
                self.settled_since = now_s
        else:
            self.settled_since = None

    def take(self, frame_stamp_s: float) -> bool:
        """Return True if a frame with this stamp is the view of the current heading."""
        return (self.active and self.settled_since is not None
                and frame_stamp_s >= self.settled_since + self.settle_s)

    def advance(self) -> Optional[float]:
        """Step to the next heading; return its target, or home when all views are in."""
        self.k += 1
        self.settled_since = None
        if self.k == self.headings:
            self.active = False
        return self.target()

    def abort(self) -> float:
        """End the scan early; return the home target (one full turn: facing forward)."""
        self.active = False
        return self.base + self.headings * self.step

    def timed_out(self, now_s: float) -> bool:
        """Return True when an active scan has run longer than timeout_s."""
        return self.active and now_s - self.t0 > self.timeout_s
