import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="ppp",
    py_modules=["ppp"],
    version="0.0.1",
    description="Prompt++, unofficial implementation of Prompt+ with bit more flavor",
    author="Simo Ryu",
    packages=find_packages(),
    entry_points={
        "console_scripts": ["ppp_train = ppp.train_ppp:main"],
    },
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
)
