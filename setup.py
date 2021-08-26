import os
import setuptools

PACKAGE_NAME = "rrc_example_package"

setuptools.setup(
    name=PACKAGE_NAME,
    version="2.0.0",
    # Packages to export
    packages=setuptools.find_packages(),
    data_files=[
        # Install "marker" file in package index
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + PACKAGE_NAME],
        ),
        # Include our package.xml file
        (os.path.join("share", PACKAGE_NAME), ["package.xml"]),
    ],
    # This is important as well
    install_requires=["setuptools"],
    zip_safe=True,
    author="Felix Widmaier",
    author_email="felix.widmaier@tue.mpg.de",
    maintainer="Felix Widmaier",
    maintainer_email="felix.widmaier@tue.mpg.de",
    description="Example package for the Real Robot Challenge Submission System.",
    license="BSD 3-clause",
    # Like the CMakeLists add_executable macro, you can add your python
    # scripts here.
    entry_points={
        "console_scripts": [
            "real_move_up_and_down = rrc_example_package.scripts.real_move_up_and_down:main",
            "sim_move_up_and_down = rrc_example_package.scripts.sim_move_up_and_down:main",
            "real_trajectory_example_with_gym = rrc_example_package.scripts.real_trajectory_example_with_gym:main",
            "sim_trajectory_example_with_gym = rrc_example_package.scripts.sim_trajectory_example_with_gym:main",
            "dice_example_with_gym = rrc_example_package.scripts.dice_example_with_gym:main",
            "sim_dice_example_with_gym = rrc_example_package.scripts.sim_dice_example_with_gym:main",
        ],
    },
)
