"""Typed spatial targets and contact-driven drawer actions for LIBERO state observations."""

import numpy as np
from scipy.spatial.transform import Rotation

from placement import PlacementSkills


class DrawerSkills(PlacementSkills):
    def relative_target(self, reference, direction, relation):
        """Resolve a tabletop location beside a fixture or a compartment inside it."""
        if direction not in {"left", "right", "front", "back"}:
            raise ValueError(f"Unknown direction: {direction}")
        if relation == "inside":
            candidates = [f"{reference}_{direction}_contain_region"]
        elif relation == "outside":
            stem = reference.rsplit("_", 1)[0]
            candidates = [n for n in self.names() if n.endswith(f"_{stem}_{direction}_region")]
        else:
            raise ValueError(f"Unknown spatial relation: {relation}")
        matches = [n for n in candidates if n in self.names()]
        if len(matches) != 1:
            raise ValueError(
                f"No unique {relation} target for {direction} of {reference}: {matches}"
            )
        return matches[0]

    def __init__(self, env, mode, fallback_source):
        super().__init__(env, mode, fallback_source)
        self.diagnostic = None
        self._site = env.robots[0].eef_site_id
        self._neutral = self._wrist()

    def _wrist(self):
        return self._env.sim.data.site_xmat[self._site].reshape(3, 3).copy()

    def _orient(self, target):
        for _ in range(35):
            error = Rotation.from_matrix(target @ self._wrist().T).as_rotvec()
            if np.linalg.norm(error) < 0.04:
                break
            self.action([0, 0, 0, *np.clip(3 * error, -1, 1), 1 if self._closed else -1])

    def _body_frame(self, name):
        root = self._env.env.get_object(name).root_body
        return self._env.sim.data.get_body_xmat(root).reshape(3, 3)

    def _place(self, item, target, slot, count):
        if (
            self.diagnostic != "rim_off"
            and "mug" in item
            and "_table_" in target
            and target.endswith("_region")
            and count == 1
        ):
            for local in ([0.028, 0.015, 0.085], [-0.028, 0.015, 0.085], [0.0, 0.031, 0.035]):
                grasp = self._body_frame(item) @ local
                self.transfer(item, target, grasp=grasp, drop=(0, 0, 0.105))
                if np.linalg.norm(self.pos(item)[:2] - self.pos(target)[:2]) < 0.038:
                    return
        super()._place(item, target, slot, count)

    def drawer(self, region, operation):
        if operation not in ("open", "close"):
            raise ValueError(f"Unknown drawer action: {operation}")
        cabinet, level, tail = region.rsplit("_", 2)
        if tail != "region" or level not in ("bottom", "middle", "top"):
            raise ValueError(f"Expected a drawer region, received {region}")
        frame = self._body_frame(cabinet)
        front = -frame[:, 1]
        side = frame[:, 0]
        up = np.array([0.0, 0.0, 1.0])
        opening = operation == "open"
        for attempt in range(2):
            center = self.pos(region)
            extension = np.dot(center - self.pos(cabinet), front)
            if (opening and extension > 0.142) or (not opening and extension < -0.012):
                break
            handle = center + front * 0.113 - side * 0.006 - up * 0.004
            self.move(handle + front * 0.14 + up * 0.09, 35)
            tilt = (1.25 if attempt == 0 else 1.65) if opening else 0.9
            roll = 0.0 if opening else np.pi / 2
            target_wrist = (
                Rotation.from_rotvec(side * tilt).as_matrix()
                @ Rotation.from_rotvec(up * roll).as_matrix()
                @ self._neutral
            )
            self.grip(False)
            self._orient(self._neutral if self.diagnostic == "tilt_off" else target_wrist)
            height = 0.008 if opening else 0.002
            start = handle + front * 0.07 + up * height
            self.move(start, 32)
            if not opening:
                self.grip(True)
            contact = handle + front * (-0.014 if opening else 0.012) + up * height
            self.move(contact, 35)
            self.grip(True)
            self.move(contact + front * (0.29 if opening else -0.20), 75 if opening else 55)
            if opening:
                self.grip(False)
            self.move(self.eef() + front * 0.075, 22)
            self._trace.append(
                [
                    region,
                    operation,
                    "travel_m",
                    round(float(np.dot(self.pos(region) - center, front)), 3),
                ]
            )
        self.move(self.eef() + up * 0.085, 20)
        self._orient(self._neutral)
