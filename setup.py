# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
long_description = "Synecdoche is a micro-library built on top of DeepMind's Haiku library that implements hypernetworks and other methods for compressed weight search and optimization."


requires = (
    [
        "jax>=0.3.0",
        "jaxlib>=0.3.0",
        "dm-haiku",
    ],
)

# This call to setup() does all the work
setup(
    name="synecdoche",
    version="0.0.2",
    description="Synecdoche: Hypernetworks for Haiku in JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kmheckel/synecdoche",
    author="Kade Heckel",
    author_email="example@email.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    packages=["synecdoche"],
    include_package_data=True,
    install_requires=requires
)
