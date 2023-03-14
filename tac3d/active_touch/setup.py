from glob import glob
from setuptools import setup

package_name = 'active_touch'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wngfra',
    maintainer_email='wngfra@gmail.com',
    description='Tactile inference and active touch exploration',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tactile_encoding = active_touch.tactile_encoding:main'
        ],
    },
)
