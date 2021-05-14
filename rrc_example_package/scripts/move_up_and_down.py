#!/usr/bin/env python3
"""Simple example on how to move the robot."""
import json
import sys
import robot_fingers
from trifinger_simulation.tasks import move_cube

from rrc_example_package.example import move_up_and_down


def main():
    # the goal is passed as JSON string
    goal_json = sys.argv[1]
    goal = json.loads(goal_json)
    print("Goal: %s" % goal)

    # create the robot frontend
    frontend = robot_fingers.TriFingerPlatformWithObjectFrontend()

    # move the robot
    move_up_and_down(frontend, move_cube.episode_length)

    # It is possible to create custom files in "/output"
    with open("/output/hello.txt", "w") as fh:
        fh.write("Hello there!\n")


if __name__ == "__main__":
    main()
