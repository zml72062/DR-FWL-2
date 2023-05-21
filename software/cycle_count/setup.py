from distutils.core import setup

setup(
    name='cycle_count',
    description='Cycle (and substructure) counting module',
    author='anonymous',
    packages=['cycle_count'],
    package_data={'cycle_count': ['find_cycle.so']}
)
