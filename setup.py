from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=["rrc_example_package"], package_dir={"": "python"}
)

setup(**d)
