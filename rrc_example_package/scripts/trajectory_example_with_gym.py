#!/usr/bin/env python3
"""Demo on how to run the robot using the Gym environment

This demo creates a RealRobotCubeTrajectoryEnv environment and runs one episode
using a dummy policy.
"""
import json
import sys

from rrc_example_package import cube_trajectory_env
from rrc_example_package.example import PointAtTrajectoryPolicy


class RandomPolicy:
    """Dummy policy which uses random actions."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation):
        return self.action_space.sample()


def main():
    # the goal is passed as JSON string
    goal_json = sys.argv[1]
    goal = json.loads(goal_json)

    env = cube_trajectory_env.RealRobotCubeTrajectoryEnv(
        goal,
        cube_trajectory_env.ActionType.POSITION,
        step_size=1,
    )

    # policy = RandomPolicy(env.action_space)
    policy = PointAtTrajectoryPolicy(env.action_space, goal)

    observation = env.reset()
    t = 0
    is_done = False
    while not is_done:
        action = policy.predict(observation, t)
        observation, reward, is_done, info = env.step(action)
        t = info["time_index"]
        print("reward:", reward)


if __name__ == "__main__":
    main()
