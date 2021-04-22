Example Package for the Real Robot Challenge
============================================

This is a basic example for a package that can be submitted to the robots of
the [Real Robot Challenge](https://real-robot-challenge.com).

It is a normal ROS2 Python package that can be build with colcon.  However,
there are a few special files in the root directory that are needed for
running/evaluating your submissions.  See the sections on the different
challenge phases below for more on this.

This example uses purely Python, however, any package type that can be built
by colcon is okay.  So you can, for example, turn it into a CMake package if you
want to build C++ code.  For more information on this, see the [ROS2
documentation](https://docs.ros.org/en/foxy/Tutorials/Creating-Your-First-ROS2-Package.html).


Challenge Simulation Phase
--------------------------

For evaluation of the simulation phase (phase 1) of the challenge, the critical
file is the `evaluate_policy.py` at the root directory of the package.  This is
what is going to be executed by `rrc_evaluate_phase1.py` (found in `scripts/`).

For more information, see XXX
FIXME: add link to challenge documentation.

`evaluate_policy.py` is only used for the simulation phase and not relevant
anymore for the later phases that use the real robot.


Challenge Real Robot Phases
---------------------------

For the challenge phases on the real robots, you need to provide the following
files at the root directory of the package such that your jobs can executed on
the robots:

- `goal.json`:  May contain some configuration for sampling random goals or a
  fixed goal (might be useful for testing/training).  See the documentation of
  the challenge tasks for more details.
- `run`:  Script that is executed when submitting the package to the robot.
  This can, for example, be a Python script or a symlink to a script somewhere
  else inside the repository.  In the given example, it is a shell script
  running a Python script via `ros2 run`.  This approach would also work for C++
  executables.  When executed, a JSON string encoding the goal is passed as
  argument (the exact structure of the goal depends on the current task).

It is important that the `run` script is executable.  For this, you need to do
two things:

1. Add a shebang line at the top of the file (e.g. `#!/usr/bin/python3` when
   using Python or `#!/bin/bash` when using bash).
2. Mark the file as executable (e.g. with `chmod a+x run`).

When inside of `run` you want to call another script using `ros2 run` (as it is
done in this example), this other script needs to fulfil the same requirements.
