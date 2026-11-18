"""Reference policy on the skill API. Used as the solvability checker for scene variants.

It is never shown to the coding agents.
"""
import math

import numpy as np


def _wrap_half_pi(a):
    """The gripper is symmetric: any yaw is equivalent to one in [-pi/2, pi/2]."""
    a = (a + math.pi / 2) % math.pi - math.pi / 2
    return a


def run(api):
    counter = api.get_counter()
    safe_z = counter["top_z"] + 0.12
    cup = api.get_cup()
    radius, height = cup["radius"], cup["height"]

    # --- open the drawer far enough to drop the cup in from above ---------
    target_open = 0.20
    drawer = api.get_drawer()
    if drawer["opening"] < target_open - 0.015:
        h = api.get_handle()["pos"]
        api.move_to([h[0], h[1], safe_z], yaw=0.0)
        api.move_to(h, speed=0.1)
        api.close_gripper()
        pull = target_open - drawer["opening"]
        api.move_to([h[0] - pull, h[1], h[2]], speed=0.08)
        api.open_gripper()
        api.move_to([h[0] - pull, h[1], safe_z])

    # --- pick the cup, fingers clear of its handle -------------------------
    cup = api.get_cup()
    cx, cy, cz = cup["pos"]
    yaw = _wrap_half_pi(cup["yaw"])
    grasp_z = cz - height / 2 + 0.55 * height
    api.move_to([cx, cy, safe_z], yaw=yaw)
    api.move_to([cx, cy, grasp_z], speed=0.1)
    api.close_gripper()
    api.move_to([cx, cy, safe_z + 0.03], speed=0.1)

    # --- choose a free spot in the reachable part of the drawer ------------
    drawer = api.get_drawer()
    ic, size = drawer["interior_center"], drawer["interior_size"]
    margin = radius + 0.025
    x_lo = ic[0] - size[0] / 2 + margin
    x_hi = min(ic[0] + size[0] / 2 - margin, counter["front_x"] - margin - 0.02)
    y_lo, y_hi = ic[1] - size[1] / 2 + margin, ic[1] + size[1] / 2 - margin
    obstacles = [o for o in api.get_objects() if o["location"] == "drawer"]
    best, best_score = None, -1.0
    for x in np.linspace(x_lo, max(x_lo, x_hi), 5):
        for y in np.linspace(y_lo, y_hi, 7):
            clear = min([math.hypot(x - o["pos"][0], y - o["pos"][1])
                         - 0.5 * math.hypot(o["size"][0], o["size"][1]) for o in obstacles],
                        default=1.0)
            score = clear - 0.3 * abs(y - ic[1]) - 0.3 * abs(x - (x_lo + x_hi) / 2)
            if score > best_score:
                best, best_score = (x, y), score
    px, py = best
    place_z = drawer["floor_z"] + 0.55 * height + 0.008
    api.move_to([px, py, safe_z + 0.03], yaw=0.0)
    api.move_to([px, py, place_z], speed=0.1)
    api.open_gripper()
    api.move_to([px, py, safe_z])

    # --- close the drawer ---------------------------------------------------
    drawer = api.get_drawer()
    h = api.get_handle()["pos"]
    api.move_to([h[0], h[1], safe_z], yaw=0.0)
    api.move_to(h, speed=0.1)
    api.close_gripper()
    api.move_to([h[0] + drawer["opening"] + 0.012, h[1], h[2]], speed=0.08)
    api.open_gripper()
    ee = api.get_ee()["pos"]
    api.move_to([ee[0], ee[1], safe_z])
