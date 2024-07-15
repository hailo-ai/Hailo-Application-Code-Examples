from setuptools import setup, find_packages
import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_hailo_package():
    try:
        import hailo
    except ImportError:
        logger.error("Hailo python package not found. Please make sure you're in the Hailo virtual environment. Run 'source setup_env.sh' and try again.")
        sys.exit(1)

def read_requirements():
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()

def run_shell_command(command, error_message):
    logger.info(f"Running command: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        logger.error(f"{error_message}. Exit code: {result.returncode}")
        sys.exit(result.returncode)

def main():
    check_hailo_package()

    requirements = read_requirements()

    logger.info("Compiling C++ code...")
    run_shell_command("./compile_postprocess.sh", "Failed to compile C++ code")

    logger.info("Downloading Resources...")
    run_shell_command("./download_resources.sh", "Failed to download resources")

    setup(
        name='clip-app',
        version='0.4',
        author='Gilad Nahor',
        author_email='giladn@hailo.ai',
        description='Real time CLIP zero shot classification and detection',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        packages=find_packages(),
        install_requires=requirements,
        entry_points={
            'console_scripts': [
                'clip_app=clip_app.clip_app:main',
                'text_image_matcher=clip_app.text_image_matcher:main',
            ],
        },
        package_data={
            'clip_app': ['*.json', '*.sh', '*.cpp', '*.hpp', '*.pc'],
        },
        include_package_data=True,
    )

if __name__ == '__main__':
    main()
