"""Physical execution of an ordered, grounded placement plan (MuJoCo OSC)."""

from collections import Counter
import numpy as np

import episode
from libero_long import Skills


class PlacementSkills(Skills):
    def __init__(self, env, mode, fallback_source):
        super().__init__(env)
        self.mode = mode
        self._fallback_source = fallback_source

    def fallback(self):
        episode.load_policy(self._fallback_source)(self)

    def nearest(self, region, candidates):
        return min(
            candidates, key=lambda item: np.linalg.norm(self.pos(item)[:2] - self.pos(region)[:2])
        )

    def arrange(self, pairs):
        if not pairs or any(
            a not in self._env.env.objects_dict or b not in self.names() or a == b for a, b in pairs
        ):
            raise ValueError(f"Invalid placement plan: {pairs}")
        counts = Counter(b for _, b in pairs)
        used = Counter()
        for item, target in pairs:
            slot = used[target]
            used[target] += 1
            if self.mode == "plan_only":
                self.transfer(item, target)
            elif self.mode in ("full", "no_spacing", "center_grasp"):
                self._place(item, target, slot, counts[target])
            else:
                raise ValueError(self.mode)

    def _place(self, item, target, slot, count):
        offsets = [(0.0, 0.0, 0.0), (0.018, 0.0, 0.015), (-0.02, 0.0, 0.0)]
        if "bowl" in item or "mug" in item:
            radius = 0.055 if "bowl" in item else 0.031
            offsets = [(0.0, radius, 0.023), (0.0, -radius, 0.023), (radius, 0.0, 0.015)]
        goal = self.pos(target)
        if self.mode == "center_grasp":
            offsets = [(0.0, 0.0, 0.0)] * len(offsets)
        if count > 1 and self.mode != "no_spacing":
            goal[0] += 0.09 * (slot / (count - 1) - 0.5)
        for delta in offsets:
            p = self.pos(item)
            grasp = p + delta
            hover = max(p[2], goal[2]) + 0.17
            self.grip(False)
            self.move([*grasp[:2], hover], 26)
            self.move(grasp, 25)
            self.grip(True)
            self.move([*grasp[:2], hover], 26)
            held = self.pos(item)
            lift = held[2] - p[2]
            self._trace.append([item, target, "lift_m", round(float(lift), 3)])
            if lift < 0.025:
                continue
            hand_offset = self.eef() - held
            place = goal + hand_offset
            place[2] += 0.09
            self.move([*place[:2], hover + 0.025], 28)
            self.move(place, 24)
            self.grip(False)
            self.move([*place[:2], hover + 0.04], 14)
            miss = float(np.linalg.norm(self.pos(item)[:2] - goal[:2]))
            self._trace.append([item, target, "placed_xy_error_m", round(miss, 3)])
            if miss < 0.045:
                return

    def holds_after_release(self):
        # Observational check after official scoring; no policy commands or retries.
        for steps, vertical in [(10, 0.0), (12, 0.4), (18, 0.0)]:
            for _ in range(steps):
                self._env.step([0, 0, vertical, 0, 0, 0, -1])
        return bool(self._env.check_success())
