#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = []

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest",
]


setup(
    author="Ellie Karanikola",
    author_email="elizkaranikola@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12.7",
        "License :: OSI Approved :: MIT License",
    ],
    description="",
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords="bayesian_clustering",
    name="bayesian_clustering",
    packages=find_packages(include=["bayesian_clustering"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="",
    version="0.1.0",
    zip_safe=False,
)
