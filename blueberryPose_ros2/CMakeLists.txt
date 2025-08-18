from setuptools import setup
import os
from glob import glob

package_name = 'yolov11_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # so ROS2 can find your package
        ('share/ament_index/resource_index/packages',
            [os.path.join('resource', package_name)]),
        ('share/' + package_name, ['package.xml']),
        # install any launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # install any RViz configs, etc.
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=[
        'setuptools',
        # add any Python deps here (e.g. ultralytics, torch, numpy, opencv-python, etc.)
    ],
    zip_safe=True,
    author='Aaron McAfee',
    author_email='aaronmcafee@example.com',
    description='YOLOv11 + keypoint‐heatmap ROS2 node for blueberries',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'detection_node = yolov11_ros2.detection_node:main',
        ],
    },
)
