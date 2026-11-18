"""Hand-written policy that overfits the twin scene on purpose (ceiling-effect pilot).

It assumes what is always true in the twin: the drawer starts closed, the cup
handle points at the robot, nothing else is in the scene, and the drawer and
counter are where the twin puts them.
"""


def run(api):
    safe_z = 0.41
    handle = [0.516, -0.088, 0.205]  # twin handle grasp point, drawer closed
    api.move_to([handle[0], handle[1], safe_z], yaw=0.0)
    api.move_to(handle, speed=0.1)
    api.close_gripper()
    api.move_to([handle[0] - 0.20, handle[1], handle[2]], speed=0.08)
    api.open_gripper()
    api.move_to([handle[0] - 0.20, handle[1], safe_z])

    cup = api.get_cup()
    x, y, z = cup["pos"]
    api.move_to([x, y, safe_z], yaw=0.0)
    api.move_to([x, y, z + 0.003], speed=0.1)
    api.close_gripper()
    api.move_to([x, y, safe_z + 0.03], speed=0.1)

    place = [0.46, -0.088, 0.118]  # fixed spot in the opened twin drawer
    api.move_to([place[0], place[1], safe_z + 0.03])
    api.move_to(place, speed=0.1)
    api.open_gripper()
    api.move_to([place[0], place[1], safe_z])

    api.move_to([handle[0] - 0.20, handle[1], safe_z])
    api.move_to([handle[0] - 0.20, handle[1], handle[2]], speed=0.1)
    api.close_gripper()
    api.move_to([handle[0] + 0.012, handle[1], handle[2]], speed=0.08)
    api.open_gripper()
    api.move_to([handle[0] + 0.012, handle[1], safe_z])
