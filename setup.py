from setuptools import find_packages, setup

setup(name="log_analyzer", packages=find_packages(
    exclude=('data', 'config', 'test', 'runs', 'notebooks')
))
