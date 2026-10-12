from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'bevnav_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
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
            'gt_tf_relay_node = bevnav_ros.gt_tf_relay_node:main',
            'gt_mask_node = bevnav_ros.gt_mask_node:main',
            'bev_node = bevnav_ros.bev_node:main',
            'costmap_node = bevnav_ros.costmap_node:main',
            'planner_node = bevnav_ros.planner_node:main',
            'controller_node = bevnav_ros.controller_node:main',
            'scan_manager_node = bevnav_ros.scan_manager_node:main',
            'fusion_node = bevnav_ros.fusion_node:main',
            'memory_node = bevnav_ros.memory_node:main',
        ],
    },
)
