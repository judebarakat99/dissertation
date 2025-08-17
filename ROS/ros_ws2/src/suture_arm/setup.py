from setuptools import find_packages, setup

package_name = 'suture_arm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f'share/{package_name}/scenes', ['resource/ur3_suture.ttt']), 
        #(f'share/{package_name}/urdf',   ['urdf/mat.urdf.xacro']), 
    ],
    install_requires=[
        'setuptools',
        'ikpy',
        'numpy',
        'transforms3d',
        'coppeliasim-zmqremoteapi-client',
    ],
    zip_safe=True,
    maintainer='mscrobotics2425laptop37',
    maintainer_email='judebarakat@yahoo.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'suturing = suture_arm.suture_arm_node:main',
        ],
    },
)
