#!/usr/bin/env python3
"""Demo on how to run the robot using the Gym environment

This demo creates a RealRobotRearrangeDiceEnv environment and runs one episode
using a dummy policy.
"""
import json
import sys

from rrc_example_package import rearrange_dice_env
from rrc_example_package.example import PointAtDieGoalPositionsPolicy


def main():
    # the goal is passed as JSON string
    goal_json = sys.argv[1]
    goal = json.loads(goal_json)

    env = rearrange_dice_env.RealRobotRearrangeDiceEnv(
        goal,
        rearrange_dice_env.ActionType.POSITION,
        step_size=1,
    )

    policy = PointAtDieGoalPositionsPolicy(env.action_space, goal)

    observation = env.reset()
    t = 0
    is_done = False
    while not is_done:
        action = policy.predict(observation, t)
        observation, reward, is_done, info = env.step(action)
        t = info["time_index"]


if __name__ == "__main__":
    main()
