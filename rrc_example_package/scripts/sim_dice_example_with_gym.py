#!/usr/bin/env python3
"""Demo on how to run the simulation using the Gym environment

This demo creates a SimRearrangeDiceEnv environment and runs one episode using
a dummy policy.
"""
from rrc_example_package import rearrange_dice_env
from rrc_example_package.example import PointAtDieGoalPositionsPolicy


def main():
    env = rearrange_dice_env.SimRearrangeDiceEnv(
        goal=None,  # passing None to sample a random goal
        action_type=rearrange_dice_env.ActionType.POSITION,
        visualization=True,
    )

    is_done = False
    observation = env.reset()
    t = 0

    policy = PointAtDieGoalPositionsPolicy(env.action_space, env.current_goal)
    while not is_done:
        action = policy.predict(observation, t)
        observation, reward, is_done, info = env.step(action)
        t = info["time_index"]


if __name__ == "__main__":
    main()
