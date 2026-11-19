"""Selectable physical grasps. Both planners use the same implementation and observations."""

import numpy as np
from scipy.spatial.transform import Rotation

from drawer_skills import DrawerSkills


class GraspSkills(DrawerSkills):
    def place_with(self, item, target, grasp):
        if item not in self._env.env.objects_dict or target not in self.names() or item == target:
            raise ValueError(f"Invalid placement {item} -> {target}")
        if grasp not in ("rim", "thin_edge", "standard"):
            raise ValueError(f"Unknown grasp: {grasp}")
        self._trace.append([item, target, "requested_grasp", grasp])
        if grasp == "standard" or self.diagnostic == "fixed":
            self.arrange([(item, target)])
            return np.linalg.norm(self.pos(item)[:2] - self.pos(target)[:2]) < 0.045
        if grasp == "rim":
            offsets = [(0.0, 0.06, 0.025), (0.0, -0.06, 0.025), (0.0, 0.05, 0.035)]
            if self.diagnostic == "old_rim_offset":
                offsets = [(0.0, 0.055, 0.023), (0.0, -0.055, 0.023), (0.055, 0.0, 0.015)]
            return self._carry(item, target, offsets, 0.08)
        return self._thin_edge(item, target)

    def _carry(self, item, target, offsets, release, geom=None):
        # Calibrated from the archived Claude rim routine. Verify lift before transport.
        up = np.array([0.0, 0.0, 1.0])
        for offset in offsets:
            before = self.pos(item)
            contact = before + offset
            self.grip(False)
            if self.diagnostic == "legacy_pick_motion":
                hover = max(before[2], self.pos(target)[2]) + 0.17
                self.move([*contact[:2], hover], 26)
                self.move(contact, 25)
                self.grip(True)
                self.move([*contact[:2], hover], 26)
            else:
                hand = self.eef()
                self.move([*hand[:2], max(hand[2], contact[2] + 0.12)], 10)
                self.move(contact + up * 0.10, 22)
                self.move(contact, 18)
                self.grip(True)
                self.move(contact + up * 0.08, 12)
            held = self.pos(item)
            lift = float(held[2] - before[2])
            self._trace.append([item, target, "lift_m", round(lift, 3)])
            if lift > 0.025:
                hand = self.eef()
                dest = self.pos(target) + up * release + hand - held
                hover = max(hand[2], dest[2] + 0.12)
                self.move([*hand[:2], hover], 15)
                if self.pos(item)[2] >= held[2] - 0.02:
                    if geom is not None:
                        return self._insert_thin(item, target, geom, hover, release)
                    self.move([*dest[:2], hover], 35)
                    self.move(dest, 20)
                    self.grip(False)
                    self.move(self.eef() + up * 0.10, 12)
                    if np.linalg.norm(self.pos(item)[:2] - self.pos(target)[:2]) < 0.045:
                        return True
            self.grip(False)
            self.move(self.eef() + up * 0.12, 15)
        return False

    def _thin_edge(self, item, target):
        # The Panda jaws close along wrist x in these LIBERO environments.
        # Measure the short horizontal axis and upper edge of the visible collider.
        obj = self._env.env.get_object(item)
        sim = self._env.sim
        boxes = [sim.model.geom_name2id(n) for n in obj.contact_geoms]
        boxes = [i for i in boxes if sim.model.geom_type[i] == 6]
        if not boxes:
            self.arrange([(item, target)])
            return np.linalg.norm(self.pos(item)[:2] - self.pos(target)[:2]) < 0.045
        geom = max(boxes, key=lambda i: np.prod(sim.model.geom_size[i]))
        rotation = sim.data.geom_xmat[geom].reshape(3, 3)
        radii = sim.model.geom_size[geom]
        axis = min(np.argsort(np.abs(rotation[2]))[:2], key=lambda j: radii[j])
        direction = rotation[:, axis]
        wrist = self._wrist()
        yaw = np.arctan2(direction[1], direction[0]) - np.arctan2(wrist[1, 0], wrist[0, 0])
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
        if self.diagnostic != "no_pick_yaw":
            self._orient(Rotation.from_euler("z", yaw).as_matrix() @ wrist)
        center = sim.data.geom_xpos[geom].copy() - self.pos(item)
        half_height = np.abs(rotation[2]) @ radii
        offsets = [center + [0, 0, half_height * f] for f in (0.45, 0.15, 0.70)]
        if self.diagnostic == "low_edge":
            for offset, height in zip(offsets, (0.0, 0.015, 0.0)):
                offset[2] = height
        return self._carry(item, target, offsets, 0.025, geom)

    def _insert_thin(self, item, target, geom, hover, release):
        # Fit an upright rectangular object through the perceived opening. The book's
        # reference frame need not be at the centre of its cover.
        sim = self._env.sim
        goal = self.pos(target)
        target_id = sim.model.site_name2id(target) if target in sim.model.site_names else None
        if (
            target_id is not None
            and "contain_region" in target
            and self.diagnostic != "no_slot_yaw"
        ):
            site_frame = sim.data.site_xmat[target_id].reshape(3, 3)
            site_size = sim.model.site_size[target_id]
            axes = np.argsort(abs(site_frame[2]))[:2]
            aperture = site_size[axes]
            body_frame = sim.data.geom_xmat[geom].reshape(3, 3)
            long_axis = max(
                np.argsort(abs(body_frame[2]))[:2], key=lambda j: sim.model.geom_size[geom, j]
            )
            length = body_frame[:, long_axis]
            phi = np.arctan2(length[1], length[0])
            axis = site_frame[:, axes[np.argmax(aperture)]]
            angle = np.arctan2(axis[1], axis[0])
            if min(aperture) / max(aperture) > 0.8:
                angle += np.pi / 4
            delta = (angle - phi + np.pi / 2) % np.pi - np.pi / 2
            self._orient(Rotation.from_euler("z", delta).as_matrix() @ self._wrist())
        for height, steps in ((hover, 45), (hover, 24), (None, 35)):
            hand = self.eef()
            dest = goal + hand - self.pos(item)
            dest[:2] = goal[:2] + hand[:2] - sim.data.geom_xpos[geom, :2]
            dest[2] = height if height is not None else dest[2] + release
            self.move(dest, steps)
        self.grip(False)
        self.move(self.eef() + [0, 0, 0.12], 20)
        return np.linalg.norm(self.pos(item)[:2] - goal[:2]) < 0.045
