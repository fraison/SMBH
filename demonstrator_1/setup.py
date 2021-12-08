
#!/usr/bin/env python
from setuptools import setup, find_packages
import os

setup(name='SMBH',
      #Standard variable in build deployment pipeline (used to automatically version the library on each build)
      #version=os.environ['BUILD_BUILDNUMBER'],
      version='1.0',
      description="",
      url='',
      author='',
      author_email='',
      # install_requires=required,
      # include_package_data=True,
      packages=find_packages())

