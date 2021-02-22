#!/usr/bin/env python3
"""Simple example on how to move the robot."""
import json
import sys
import robot_fingers
from trifinger_simulation.tasks import move_cube

from rrc_example_package.example import move_up_and_down


def main():
    # the difficulty level and the goal pose (as JSON string) are passed as
    # arguments
    difficulty = int(sys.argv[1])
    goal_pose_json = sys.argv[2]
    goal = json.loads(goal_pose_json)
    print(
        "Goal: %s/%s (difficulty: %d)"
        % (goal["position"], goal["orientation"], difficulty)
    )

    # create the robot frontend
    frontend = robot_fingers.TriFingerPlatformFrontend()

    # move the robot
    move_up_and_down(frontend, 20000)

    # It is possible to create custom files in "/output"
    with open("/output/hello.txt", "w") as fh:
        fh.write("Hello there!\n")


if __name__ == "__main__":
    main()
