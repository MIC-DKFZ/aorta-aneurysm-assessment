"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

from setuptools import setup, find_packages


setup(
    name="aorta-aneurysm",
    version="1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'aorta_aneurysm_v1=aorta_aneurysm.__main__:main'
        ]
    }
)