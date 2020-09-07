#!/usr/bin/env python3
"""Simple example on how to move the robot."""
import json
import sys
import numpy as np
import robot_interfaces
import robot_fingers


# Number of actions in one episode (1000 actions per second for two minutes)
episode_length = 2 * 60 * 1000


def move_up_and_down(frontend: robot_fingers.TriFingerPlatformFrontend):
    """Move up and down multiple times using fixed goal positions."""
    position_down = [-0.08, 0.84, -1.2] * 3
    position_up = [0.5, 1.2, -2.4] * 3
    target_positions = [position_down, position_up]

    i = 0
    while True:
        print("Iteration {}".format(i))
        action = robot_interfaces.trifinger.Action(position=target_positions[i % 2])
        i += 1

        for _ in range(500):
            t = frontend.append_desired_action(action)
            frontend.wait_until_timeindex(t)

            # make sure to not exceed the number of allowed actions
            if t >= episode_length - 1:
                return

        observation = frontend.get_robot_observation(t)
        print("Finger positions: ", observation.position)


def main():
    # the difficulty level and the goal pose (as JSON string) are passed as
    # arguments
    difficulty = int(sys.argv[1])
    goal_pose_json = sys.argv[2]
    goal = json.loads(goal_pose_json)
    print("Goal: %s (difficulty: %d)" % (goal_pose_json, difficulty))

    # create the robot frontend
    frontend = robot_fingers.TriFingerPlatformFrontend()

    # move the robot
    move_up_and_down(frontend)

    # It is possible to create custom files in "/output"
    with open("/output/hello.txt", "w") as fh:
        fh.write("Hello there!\n")


if __name__ == "__main__":
    main()
