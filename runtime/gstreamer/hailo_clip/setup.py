from setuptools import setup, find_packages
import os
import sys

try:
    import hailo
except ImportError:
    print("Hailo python package not found. Please make sure you're in the Hailo virtual environment. Run 'source setup_env.sh' and try again.")
    exit(1)

# Read requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# Compile C++ code using Meson
print("Compiling C++ code...")
exit_code = os.system("./compile_postprocess.sh")
if exit_code != 0:
    sys.exit(f"Failed to compile C++ code. Exit code: {exit_code}")

# Download HEF and videos to the resources folder
print("Downloading Resources...")
exit_code = os.system("./download_resources.sh")
if exit_code != 0:
    sys.exit(f"Failed to download resources. Exit code: {exit_code}")

# Setup function
setup(
    name='clip-app',
    version='0.3',
    author='Gilad Nahor',
    author_email='giladn@hailo.ai',
    description='Real time clip classification and detection',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'clip_app=clip_app.clip_app:main',
            'text_image_matcher = clip_app.TextImageMatcher:main',
        ],
    },
    scripts=[
        'compile_postprocess.sh',
        'download_resources.sh'
    ],
    package_data={
        'clip_app': ['*.json', '*.sh', '*.cpp', '*.hpp', '*.pc'],
    },
    include_package_data=True,
)