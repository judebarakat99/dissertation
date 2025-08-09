from setuptools import setup

package_name = 'suture_arm'
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
                (f'share/{package_name}', ['package.xml'])],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Suturing driver (ROS 2 + CoppeliaSim, no MoveIt)',
    entry_points={'console_scripts': [
        'suturing = suture_arm.suture_arm_node:main',
    ]},
)
