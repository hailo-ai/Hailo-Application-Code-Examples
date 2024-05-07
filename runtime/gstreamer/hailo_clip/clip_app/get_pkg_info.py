import subprocess
import os

# This function gets the tappas_workspace and version to be used by the application
# It first checks if the environment variables TAPPAS_WORKSPACE and TAPPAS_VERSION are set
# If they are not set, it uses pkg-config to get the data from the pkg-config file
# If the pkg-config file is not accessible, it raises an exception
# Make sure /opt/hailo/tappas/pkgconfig/hailo_tappas.pc is accessible and has the right owner and permissions
# Make sure PKG_CONFIG_PATH include this directory
# export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/opt/hailo/tappas/pkgconfig

def get_pkg_info(package_name="hailo_tappas"):
    info = {}
    try:
        # Get the tappas_workspace environmet variable
        TAPPAS_WORKSPACE = os.environ.get("TAPPAS_WORKSPACE", "")
        if TAPPAS_WORKSPACE == "":
            # Get the tappas_workspace variable
            tappas_workspace = subprocess.check_output(['pkg-config', '--variable=tappas_workspace', package_name], text=True).strip()
            info['tappas_workspace'] = tappas_workspace
        else:
            info['tappas_workspace'] = TAPPAS_WORKSPACE
        
        # Get the version
        TAPPAS_VERSION = os.environ.get("TAPPAS_VERSION", "")
        if TAPPAS_VERSION == "":
            version = subprocess.check_output(['pkg-config', '--modversion', package_name], text=True).strip()
            info['version'] = version
        else:
            info['version'] = TAPPAS_VERSION

    except subprocess.CalledProcessError:
        raise Exception("TAPPAS_WORKSPACE OR TAPPAS_VERSION environment variables are not set. Could not get data from pkg-config. Make sure /opt/hailo/tappas/pkgconfig/hailo_tappas.pc is accessible.")
    print(f"Using TAPPAS_WORKSPACE: {info['tappas_workspace']} and TAPPAS version: {info['version']}")
    return info

if __name__ == "__main__":
    package_name = "hailo_tappas"  # Replace with your package name
    info = get_pkg_info(package_name)
    if info is not None:
        print(info)
