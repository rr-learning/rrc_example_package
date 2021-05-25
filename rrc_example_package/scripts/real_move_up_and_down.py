#!/usr/bin/env python3
"""Simple example on how to move the robot."""
import json
import sys
import robot_interfaces
import robot_fingers
from trifinger_simulation.tasks import move_cube_on_trajectory


def main():
    # the goal is passed as JSON string
    goal_json = sys.argv[1]
    goal = json.loads(goal_json)
    print("Goal: %s" % goal)

    # It is possible to create custom files in "/output"
    with open("/output/hello.txt", "w") as fh:
        fh.write("Hello there!\n")

    # create the robot frontend
    frontend = robot_fingers.TriFingerPlatformWithObjectFrontend()

    # move the robot between two fixed positions ("up" and "down")
    position_down = [-0.08, 0.84, -1.2] * 3
    position_up = [0.5, 1.2, -2.4] * 3
    target_positions = [position_down, position_up]

    i = 0
    while True:
        print("Iteration {}".format(i))
        action = robot_interfaces.trifinger.Action(
            position=target_positions[i % 2]
        )
        i += 1

        for _ in range(500):
            t = frontend.append_desired_action(action)
            frontend.wait_until_timeindex(t)

            # make sure to not exceed the number of allowed actions
            if t >= move_cube_on_trajectory.EPISODE_LENGTH - 1:
                return

        robot_observation = frontend.get_robot_observation(t)
        print("Finger positions:", robot_observation.position)

        camera_observation = frontend.get_camera_observation(t)
        print("Object position:", camera_observation.object_pose.position)
