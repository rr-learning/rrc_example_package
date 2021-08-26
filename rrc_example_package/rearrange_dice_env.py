"""Example Gym environment for the RRC 2021 Phase 2."""
import enum
import pathlib
import typing

import gym
import numpy as np
import cv2

import robot_fingers
import trifinger_simulation
import trifinger_simulation.tasks.rearrange_dice as task
from trifinger_simulation import trifingerpro_limits
from trifinger_simulation.camera import (
    load_camera_parameters,
    CameraParameters,
)
from trifinger_simulation.trifinger_platform import ObjectType
from trifinger_cameras.utils import convert_image
from trifinger_object_tracking.py_lightblue_segmenter import segment_image


CONFIG_DIR = pathlib.Path("/etc/trifingerpro")


class ActionType(enum.Enum):
    """Different action types that can be used to control the robot."""

    #: Use pure torque commands.  The action is a list of torques (one per
    #: joint) in this case.
    TORQUE = enum.auto()
    #: Use joint position commands.  The action is a list of angular joint
    #: positions (one per joint) in this case.  Internally a PD controller is
    #: executed for each action to determine the torques that are applied to
    #: the robot.
    POSITION = enum.auto()
    #: Use both torque and position commands.  In this case the action is a
    #: dictionary with keys "torque" and "position" which contain the
    #: corresponding lists of values (see above).  The torques resulting from
    #: the position controller are added to the torques in the action before
    #: applying them to the robot.
    TORQUE_AND_POSITION = enum.auto()


class RealRobotRearrangeDiceEnv(gym.GoalEnv):
    """Gym environment for rearranging dice with a TriFingerPro robot."""

    def __init__(
        self,
        goal: typing.Optional[task.Goal] = None,
        action_type: ActionType = ActionType.POSITION,
        step_size: int = 1,
    ):
        """Initialize.

        Args:
            goal: Goal pattern for the dice.  If ``None`` a new random goal is
                sampled upon reset.
            action_type: Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size:  Number of actual control steps to be performed in one
                call of step().
        """
        # Basic initialization
        # ====================

        if goal is not None:
            task.validate_goal(goal)
        self.goal = goal

        self.action_type = action_type

        if step_size < 1:
            raise ValueError("step_size cannot be less than 1.")
        self.step_size = step_size

        # will be initialized in reset()
        self.platform = None

        # load camera parameters
        self.camera_params = load_camera_parameters(
            CONFIG_DIR, "camera{id}_cropped_and_downsampled.yml"
        )

        # Create the action and observation spaces
        # ========================================

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )

        mask_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 270, 270), dtype=np.uint8
        )

        if self.action_type == ActionType.TORQUE:
            self.action_space = robot_torque_space
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self.action_space = robot_position_space
            self._initial_action = trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": robot_torque_space,
                    "position": robot_position_space,
                }
            )
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")

        self.observation_space = gym.spaces.Dict(
            {
                "robot_observation": gym.spaces.Dict(
                    {
                        "position": robot_position_space,
                        "velocity": robot_velocity_space,
                        "torque": robot_torque_space,
                    }
                ),
                "action": self.action_space,
                "desired_goal": mask_space,
                "achieved_goal": mask_space,
            }
        )

    def compute_reward(
        self,
        achieved_goal: typing.Sequence[np.ndarray],
        desired_goal: typing.Sequence[np.ndarray],
        info: dict,
    ) -> float:
        """Compute the reward for the given achieved and desired goal.

        Args:
            achieved_goal: Segmentation mask of the observed camera images.
            desired_goal: Segmentation mask of the goal positions.
            info: Unused.

        Returns:
            The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal. Note that the following should always hold true::

                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(
                    ob['achieved_goal'],
                    ob['desired_goal'],
                    info,
                )
        """
        return -task.evaluate_state(desired_goal, achieved_goal)

    def seed(self, seed=None):
        """Sets the seed for this env’s random number generator.

        .. note::

           Spaces need to be seeded separately.  E.g. if you want to sample
           actions directly from the action space using
           ``env.action_space.sample()`` you can set a seed there using
           ``env.action_space.seed()``.

        Returns:
            List of seeds used by this environment.  This environment only uses
            a single seed, so the list contains only one element.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        task.seed(seed)
        return [seed]

    def _create_observation(self, t, action):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)

        segmentation_masks = [
            segment_image(convert_image(c.image))
            for c in camera_observation.cameras
        ]

        observation = {
            "robot_observation": {
                "position": robot_observation.position,
                "velocity": robot_observation.velocity,
                "torque": robot_observation.torque,
            },
            "action": action,
            "desired_goal": self.goal_masks,
            "achieved_goal": segmentation_masks,
        }
        return observation

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        Important: ``reset()`` needs to be called before doing the first step.

        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).

        Returns:
            tuple:

            - observation (dict): agent's observation of the current
              environment.
            - reward (float): amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the current time index.
        """
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.step_size

        # ensure episode length is not exceeded due to step_size
        step_count_after = self.info["time_index"] + num_steps
        if step_count_after > task.EPISODE_LENGTH:
            excess = step_count_after - task.EPISODE_LENGTH
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            self.info["time_index"] = t

            observation = self._create_observation(t, action)

            reward += self.compute_reward(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.info,
            )

            # make sure to not exceed the episode length
            if t >= task.EPISODE_LENGTH - 1:
                break

        is_done = t >= task.EPISODE_LENGTH

        return observation, reward, is_done, self.info

    def reset(self):
        # cannot reset multiple times
        if self.platform is not None:
            raise RuntimeError(
                "Once started, this environment cannot be reset."
            )

        self.platform = robot_fingers.TriFingerPlatformFrontend()

        # if no goal is given, sample one randomly
        if self.goal is None:
            goal = task.sample_goal()
        else:
            goal = self.goal

        self.goal_masks = task.generate_goal_mask(self.camera_params, goal)

        self.info = {"time_index": -1}

        # need to already do one step to get initial observation
        # TODO disable frameskip here?
        observation, _, _, _ = self.step(self._initial_action)

        return observation


class SimRearrangeDiceEnv(gym.GoalEnv):
    """Gym environment for rearranging dice with a TriFingerPro robot."""

    def __init__(
        self,
        goal: typing.Optional[task.Goal] = None,
        action_type: ActionType = ActionType.POSITION,
        step_size: int = 1,
        visualization: bool = True,
    ):
        """Initialize.

        Args:
            goal: Goal pattern for the dice.  If ``None`` a new random goal is
                sampled upon reset.
            action_type: Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size:  Number of actual control steps to be performed in one
                call of step().
        """
        # Basic initialization
        # ====================

        if goal is not None:
            task.validate_goal(goal)
        self.goal = goal

        self.action_type = action_type

        if step_size < 1:
            raise ValueError("step_size cannot be less than 1.")
        self.step_size = step_size

        self.visualization = visualization

        # will be initialized in reset()
        self.platform = None

        # Set camera parameters as used in simulation
        # view 1
        pose_60 = (
            np.array(
                (
                    -0.6854993104934692,
                    -0.5678349733352661,
                    0.45569100975990295,
                    0.0,
                    0.7280372381210327,
                    -0.5408401489257812,
                    0.4212528169155121,
                    0.0,
                    0.007253906223922968,
                    0.6205285787582397,
                    0.7841504216194153,
                    0.0,
                    -0.01089033205062151,
                    0.014668643474578857,
                    -0.5458434820175171,
                    1.0,
                )
            )
            .reshape(4, 4)
            .T
        )
        # view 2
        pose_180 = (
            np.array(
                (
                    0.999718189239502,
                    0.02238837257027626,
                    0.007906466722488403,
                    0.0,
                    -0.01519287470728159,
                    0.8590874671936035,
                    -0.5116034150123596,
                    0.0,
                    -0.01824631541967392,
                    0.5113391280174255,
                    0.8591853380203247,
                    0.0,
                    -0.000687665306031704,
                    0.01029178500175476,
                    -0.5366422533988953,
                    1.0,
                )
            )
            .reshape(4, 4)
            .T
        )
        # view 3
        pose_300 = (
            np.array(
                (
                    -0.7053901553153992,
                    0.5480064153671265,
                    -0.44957074522972107,
                    0.0,
                    -0.7086654901504517,
                    -0.5320233702659607,
                    0.4634052813053131,
                    0.0,
                    0.014766914770007133,
                    0.6454768180847168,
                    0.7636371850967407,
                    0.0,
                    -0.0019663232378661633,
                    0.0145435631275177,
                    -0.5285998582839966,
                    1.0,
                )
            )
            .reshape(4, 4)
            .T
        )
        # proj
        pb_proj = (
            np.array(
                (
                    2.0503036975860596,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0503036975860596,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0000200271606445,
                    -1.0,
                    0.0,
                    0.0,
                    -0.002000020118430257,
                    0.0,
                )
            )
            .reshape(4, 4)
            .T
        )
        width = 270
        height = 270
        x_scale = pb_proj[0, 0]
        y_scale = pb_proj[1, 1]
        c_x = width / 2
        c_y = height / 2
        f_x = x_scale * c_x
        f_y = y_scale * c_y
        camera_matrix = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 0]])

        dist = (0, 0, 0, 0, 0)

        self.camera_params = (
            CameraParameters(
                "camera60", width, height, camera_matrix, dist, pose_60
            ),
            CameraParameters(
                "camera180", width, height, camera_matrix, dist, pose_180
            ),
            CameraParameters(
                "camera300", width, height, camera_matrix, dist, pose_300
            ),
        )

        # Create the action and observation spaces
        # ========================================

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )

        mask_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 270, 270), dtype=np.uint8
        )

        if self.action_type == ActionType.TORQUE:
            self.action_space = robot_torque_space
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self.action_space = robot_position_space
            self._initial_action = trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": robot_torque_space,
                    "position": robot_position_space,
                }
            )
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")

        self.observation_space = gym.spaces.Dict(
            {
                "robot_observation": gym.spaces.Dict(
                    {
                        "position": robot_position_space,
                        "velocity": robot_velocity_space,
                        "torque": robot_torque_space,
                    }
                ),
                "action": self.action_space,
                "desired_goal": mask_space,
                "achieved_goal": mask_space,
            }
        )

    def compute_reward(
        self,
        achieved_goal: typing.Sequence[np.ndarray],
        desired_goal: typing.Sequence[np.ndarray],
        info: dict,
    ) -> float:
        """Compute the reward for the given achieved and desired goal.

        Args:
            achieved_goal: Segmentation mask of the observed camera images.
            desired_goal: Segmentation mask of the goal positions.
            info: Unused.

        Returns:
            The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal. Note that the following should always hold true::

                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(
                    ob['achieved_goal'],
                    ob['desired_goal'],
                    info,
                )
        """
        return -task.evaluate_state(desired_goal, achieved_goal)

    def seed(self, seed=None):
        """Sets the seed for this env’s random number generator.

        .. note::

           Spaces need to be seeded separately.  E.g. if you want to sample
           actions directly from the action space using
           ``env.action_space.sample()`` you can set a seed there using
           ``env.action_space.seed()``.

        Returns:
            List of seeds used by this environment.  This environment only uses
            a single seed, so the list contains only one element.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        task.seed(seed)
        return [seed]

    def _create_observation(self, t, action):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)

        segmentation_masks = [
            segment_image(cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR))
            for c in camera_observation.cameras
        ]

        observation = {
            "robot_observation": {
                "position": robot_observation.position,
                "velocity": robot_observation.velocity,
                "torque": robot_observation.torque,
            },
            "action": action,
            "desired_goal": self.goal_masks,
            "achieved_goal": segmentation_masks,
        }
        return observation

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        Important: ``reset()`` needs to be called before doing the first step.

        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).

        Returns:
            tuple:

            - observation (dict): agent's observation of the current
              environment.
            - reward (float): amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the current time index.
        """
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.step_size

        # ensure episode length is not exceeded due to step_size
        step_count_after = self.step_count + num_steps
        if step_count_after > task.EPISODE_LENGTH:
            excess = step_count_after - task.EPISODE_LENGTH
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            self.step_count += 1
            if self.step_count > task.EPISODE_LENGTH:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            self.info["time_index"] = t

            observation = self._create_observation(
                self.info["time_index"], action
            )

            reward += self.compute_reward(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.info,
            )

        is_done = self.step_count >= task.EPISODE_LENGTH

        return observation, reward, is_done, self.info

    def reset(self):
        # hard-reset simulation
        del self.platform

        # initialize simulation
        initial_robot_position = trifingerpro_limits.robot_position.default

        self.platform = trifinger_simulation.TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=initial_robot_position,
            enable_cameras=True,
            object_type=ObjectType.DICE,
        )

        # if no goal is given, sample one randomly
        if self.goal is None:
            self.current_goal = task.sample_goal()
        else:
            self.current_goal = self.goal

        self.goal_masks = task.generate_goal_mask(
            self.camera_params, self.current_goal
        )

        self.info = {"time_index": -1}

        self.step_count = 0

        return self._create_observation(0, self._initial_action)
