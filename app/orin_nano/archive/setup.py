from setuptools import setup, find_packages

setup(
    name="camera_service",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask",
        "requests",
        "opencv-python",
        "numpy",
        "face_recognition",
        "flask_cors",
    ],
    description="Jetson Camera Service with facial recognition and gesture detection",
    author="MMonent",
    author_email="info@mmoment.xyz",
) 