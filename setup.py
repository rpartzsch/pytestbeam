from setuptools import setup
from setuptools import find_packages

author = "Rasmus Partzsch"

# Requirements
install_requires = [
    "numpy",
    "coloredlogs",
    "tables",
    "tqdm",
    "numba_progress",
    "matplotlib",
    "cmcrameri",
    "pyyaml",
    "scipy",
]

version = "0.9"

setup(
    name="pytestbeam",
    version=version,
    description="Pytestbeam",
    license="License AGPL-3.0 license",
    long_description=".",
    author=author,
    maintainer=author,
    install_requires=install_requires,
    python_requires=">3.10",
    packages=find_packages(),
    include_package_data=True,
    platforms="posix",
)
