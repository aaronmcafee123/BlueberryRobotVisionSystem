from setuptools import setup, find_packages

package_name = 'blueberryPose_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aaronmcafee',
    maintainer_email='aaronmcafee21@gmail.com',
    description='YOLOv12 + keypoint-heatmap ROS 2 detection node for blueberries',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'detection_node = blueberryPose_ros2.detection_node:main',
        ],
    },
)
