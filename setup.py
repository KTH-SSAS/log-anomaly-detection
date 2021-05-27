from setuptools import setup, find_packages

setup(name="log_analyzer", packages=find_packages(
    exclude=('data', 'config', 'test', 'runs', 'notebooks')
    ))
