# Copyright (c) 2019 Takahiro Itazuri
# Released under the MIT license
# https://github.com/takahiro-itazuri/torchattack/blob/master/LICENSE

import os
from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), 'torchattack/VERSION')) as f:
    version = f.read()

setup(
    name="torchattack",
    version=version,
    description="Adversarial Attacks in PyTorch",
    url="https://github.com/takahiro-itazuri/torchattack",
    author="Takahiro Itazuri",
    author_email="takahiro.lab.1226@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    license="MIT"
)
