#!/usr/bin/env python

from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='Plant Nutrition Scheduler',
    version='1.0',
    description='Plant Nutrition Analysis',
    author='Matias Chavez',
    author_email='matias.chavez@mail.ch',
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    license="MIT",  
    url="https://github.com/chavias/plant-nutrition",  
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  
    test_suite="tests",
)
