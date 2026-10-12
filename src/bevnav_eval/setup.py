from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'bevnav_eval'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config', 'scenarios'),
            glob('config/scenarios/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bluegoose',
    maintainer_email='bluegoose@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'make_gt_grid = bevnav_eval.make_gt_grid:main',
            'bev_range_check = bevnav_eval.bev_range_check:main',
            'run_trials = bevnav_eval.run_trials:main',
            'analyze_bags = bevnav_eval.analyze_bags:main',
            'make_figures = bevnav_eval.make_figures:main',
            'scan_check = bevnav_eval.scan_check:main',
        ],
    },
)
