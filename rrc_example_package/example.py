"""Simple example on how to move the robot."""
import robot_interfaces
import robot_fingers


def move_up_and_down(
    frontend: robot_fingers.TriFingerPlatformFrontend, episode_length: int
):
    """Move up and down multiple times using fixed goal positions.

    Args:
        frontend:  Frontend of the TriFingerPro platform.  Used to control the
            robot.
        episode_length:  Number of time steps in the episode.  Used to ensure
            that the limit is not exceeded.
    """
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
            if t >= episode_length - 1:
                return

        robot_observation = frontend.get_robot_observation(t)
        print("Finger positions:", robot_observation.position)

        camera_observation = frontend.get_camera_observation(t)
        print("Object position:", camera_observation.object_pose.position)
