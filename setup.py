from setuptools import setup, find_packages

setup(name='bark',
      version='0.1',
      description='tools for reading and writing BARK formatted data',
      url='http://github.com/kylerbrown/bark',
      author='Kyler Brown',
      author_email='kylerjbrown@gmail.com',
      license='GPL',
      packages=find_packages(),
      zip_safe=False,
      entry_points = {
          'console_scripts' : [
              'bark-root=bark.barkutils:mk_root',
              'bark-entry=bark.barkutils:mk_entry',
              ]
          }
      )
