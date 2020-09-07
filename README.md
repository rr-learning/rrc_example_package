Example Package for the Real Robot Challenge Submission System
==============================================================


This is a basic example for a package that can be submitted to the robots of
the Real Robot Challenge.

It is a normal catkin package.  The only special things are the following files
which are used by the robot cluster system:

- `goal.json`:  Contains the desired difficulty level and optionally a specific
  goal.  In the given example, the goal is "commented out" by changing the key
  name.  If no goal is given, a random one is sampled based on the specified
  difficulty level.  Note that the difficulty level is always needed, also if a
  specific goal is given, as it is needed for computing the reward.
- `run`:  Script that is executed when submitting the package to the robot.
  This can, for example, be a Python script or a symlink to a script somewhere
  else inside the repository.  In the given example, it is a shell script
  running a Python script via `rosrun`.  This approach would also work for C++
  executables.  When executed, the difficulty level and the goal pose of the
  object are passed as arguments.

This example uses purely Python but you could also add C++ code.  In this case
you need to add the corresponding build commands in `CMakeLists.txt` (and
potentially specify dependencies in `package.xml` as for any catkin package.
