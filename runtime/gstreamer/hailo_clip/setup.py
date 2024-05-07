from setuptools import setup, find_packages
import subprocess
import os

try:
    import hailo
except ImportError:
    print("Hailo python package found. Please make sure you're in the Hailo virtual environment. run 'source setup_env.sh' and try again.")
    exit(1)

# Read requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# Compile C++ code using Meson
# Ensure these scripts are executable and appropriately handle their execution environments.
try:
    subprocess.run(["./compile_postprocess.sh"])
except Exception as e:
    print(f"Failed to compile C++ code: {e}")

# Check if hef files exist on the resources folder and download if not
# Ensure this script is executable and appropriately handles its execution environment.
if not os.path.isfile("resources/yolov5s_personface.hef"):
    print("Downloading hef files...")
    subprocess.run(["./download_hef.sh"])

# Setup function
setup(
    name='clip-app',
    version='0.2',
    author='Gilad Nahor',
    author_email='giladn@hailo.ai',
    description='Real time clip classication and detection',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    #url='https://github.com/yourusername/your-repo',  # Optional: project home page, if any
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
        'download_hef.sh'
    ],
    package_data={
        # Include any additional files specified here
        'clip_app': ['*.json', '*.sh', '*.cpp', '*.hpp', '*.pc'],
    },
    # Ensure that non-python data files are included in your package
    include_package_data=True,
)

