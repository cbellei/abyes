from setuptools import setup

setup(name='abyes',
      version='0.1.0',
      description='AB Testing using Bayesian statistics',
      url='http://github.com/storborg/funniest',
      author='Claudio Bellei',
      author_email='',
      license='OSI Approved Apache Software License',
      packages=['abyes'],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose']
      )
