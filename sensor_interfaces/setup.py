from setuptools import setup

package_name = 'sensor_interfaces'

setup(
    name=package_name,
    version='0.1.0',
    python_requires='>=3.10',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='wngfra',
    author_email='wngfra@gmail.com',
    description='TODO: Package description',
    license='LGPLv2.1',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tactile_interface = sensor_interfaces.TactileInterface:main'
        ],
    },
)
