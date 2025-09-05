from setuptools import find_packages, setup

package_name = 'suture_arm'

model_files = [
    'ML_detection/cuts_detector_best.pth',
    'ML_detection/cuts_detector_best2.pth',
    'ML_detection/mask_cuts_detector.pth',
    'ML_detection/mask_cuts_detector2.pth',
    'ML_detection/mask_cuts_detector3.pth',
    'ML_detection/cuts_maskrcnn_best.pth',
]

ml_files = [
    'ML_detection/cuts_detector_best.py',
    'ML_detection/cuts_detector_best2.py',
    'ML_detection/mask_cuts_detector.py',
    'ML_detection/mask_cuts_detector2.py',
    'ML_detection/mask_cuts_detector3.py',
    'ML_detection/cuts_maskrcnn_best.py',
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f'share/{package_name}/scenes', ['resource/ur3_suture.ttt']), 
        (f'share/{package_name}/launch', ['launch/suture_demo.launch.py']),
        #(f'share/{package_name}/urdf',   ['urdf/mat.urdf.xacro']), 
        (f'share/{package_name}/ml', model_files),  
        (f'share/{package_name}/ml', ml_files),
        ('share/' + package_name + '/templates', ['templates/index.html']),
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
    description='Suture arm demo',
    license='',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'suturing = suture_arm.suture_arm_node:main',
            'vision_web = suture_arm.vision_web:main', 
            'coppelia_runner = suture_arm.coppelia_runner:main',   
            'dataset_capture = suture_arm.dataset_capture:main', 
            'dataset_capture2 = suture_arm.dataset_capture2:main', 
            'stitching = suture_arm.stitching:main',
            'suture_arm_node = suture_arm.suture_arm_node:main',
            'suture_motion_node = suture_arm.suture_motion_node:main',
            'suture_entry_motion_node = suture_arm.suture_entry_motion_node:main',
            'csim_frames = suture_arm.csim_frames:main',
            'model_interface = suture_arm.model_interface:main',
            'image_to_map_mapper = suture_arm.image_to_map_mapper:main',
            'suture_executor = suture_arm.suture_executor:main',
            
        ],
    },
)
