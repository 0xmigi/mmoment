from setuptools import setup, find_packages

setup(
    name="camera_service",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "Flask==2.3.3",
        "flask-cors==4.0.0",
        "picamera2==0.3.16",
        "opencv-python==4.8.1.78",
    ],
)