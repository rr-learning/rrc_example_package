#!/usr/bin/env python3
"""Simple example on how to move the robot."""
import trifinger_simulation
from trifinger_simulation.tasks import move_cube_on_trajectory
from trifinger_simulation.trifinger_platform import ObjectType


def main():
    """Example on how to move the robot in simulation."""
    # create the robot platform simulation
    robot = trifinger_simulation.TriFingerPlatform(
        visualization=True,
        object_type=ObjectType.COLORED_CUBE,
    )

    # move the robot
    position_down = [-0.08, 0.84, -1.2] * 3
    position_up = [0.5, 1.2, -2.4] * 3
    target_positions = [position_down, position_up]

    i = 0
    while True:
        action = robot.Action(
            position=target_positions[i % 2]
        )
        i += 1

        for _ in range(500):
            t = robot.append_desired_action(action)

            # make sure to not exceed the number of allowed actions
            if t >= move_cube_on_trajectory.EPISODE_LENGTH - 1:
                return

        robot_observation = robot.get_robot_observation(t)
        print("Finger positions:", robot_observation.position)

        camera_observation = robot.get_camera_observation(t)
        print("Object position:", camera_observation.object_pose.position)
