import os
from setuptools import setup, find_packages, Command

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

setup(name='superbit-lensing',
      version='1.0',
      packages=find_packages(exclude=('tests', 'docs')),
      cmdclass={
        'clean': CleanCommand,
          }
      )
