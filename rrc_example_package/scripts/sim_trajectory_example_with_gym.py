#!/usr/bin/env python3
"""Demo on how to run the simulation using the Gym environment

This demo creates a SimCubeTrajectoryEnv environment and runs one episode using
a dummy policy.
"""
from rrc_example_package import cube_trajectory_env
from rrc_example_package.example import PointAtTrajectoryPolicy


def main():
    env = cube_trajectory_env.SimCubeTrajectoryEnv(
        goal_trajectory=None,  # passing None to sample a random trajectory
        action_type=cube_trajectory_env.ActionType.POSITION,
        visualization=True,
    )

    is_done = False
    observation = env.reset()
    t = 0

    policy = PointAtTrajectoryPolicy(env.action_space, env.info["trajectory"])
    while not is_done:
        action = policy.predict(observation, t)
        observation, reward, is_done, info = env.step(action)
        t = info["time_index"]


if __name__ == "__main__":
    main()
