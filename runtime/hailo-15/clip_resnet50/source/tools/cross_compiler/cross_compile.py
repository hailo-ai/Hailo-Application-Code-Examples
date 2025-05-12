#!/usr/bin/env python3
import argparse
import logging
import sys
from common import Arch, Target, install_compilers_apt_packages, MesonInstaller, FOLDER_NAME
POSSIBLE_BUILD_TYPES = ['debug', 'release']
class MedialibTestCasesInstaller(MesonInstaller):
    def __init__(self, arch, build_type, targets, toolchain_dir_path, remote_machine_ip=None,
                 clean_build_dir=False, install_to_toolchain_rootfs=False, skip_ut=False, extra_flags=None):        
        super().__init__(arch=arch, build_type=build_type, src_build_dir= FOLDER_NAME / '../../../source/',
                         toolchain_dir_path=toolchain_dir_path, remote_machine_ip=remote_machine_ip,
                         clean_build_dir=clean_build_dir, install_to_toolchain_rootfs=install_to_toolchain_rootfs)
        self._targets = targets
        self._skip_ut = skip_ut
        self._extra_flags = extra_flags or list()
    def get_meson_build_command(self):
        extra_flags_as_list = [f'-D{flag}={value}' for flag, value in self._extra_flags]
        # Meson automatically detect and uses the cross_file
        build_cmd = ['meson', str(self._output_build_dir), '--buildtype', self._build_type,
                     '-Dprefix=/usr',] + extra_flags_as_list
        return build_cmd
def parse_args():
    def parse_key_value(kv_string):
        try:
            key, value = kv_string.split('=')
            return key, value
        except ValueError:
            raise argparse.ArgumentTypeError("Key-value pair should be in the format key=value")
    parser = argparse.ArgumentParser(description='Cross-compile Media Library Test Cases.')
    parser.add_argument('build_type', choices=POSSIBLE_BUILD_TYPES, help='Build and compilation type')
    parser.add_argument('toolchain_dir_path', help='Toolchain directory path')
    parser.add_argument('--targets', type=Target, choices=list(Target), nargs="+", default=["all"],
                        help='Target cases to compile (default all)')
    parser.add_argument('--clean-build-dir', action='store_true', help='Delete previous build dir (default false)',
                        default=False)
    parser.add_argument('--install-to-toolchain-rootfs', action='store_true', help='Install to toolchain rootfs (default false)',
                        default=False)
    parser.add_argument('--remote-machine-ip', help='remote machine ip')
    parser.add_argument('--skip-ut', action='store_true', help='Skip unit tests (default false)', default=False)
    parser.add_argument('--extra-flag', action='append', type=parse_key_value,
                        metavar='key=value', help='extra compilation flags, key-value pairs')
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    arch = Arch.ARMV8A
    install_compilers_apt_packages(arch)
    ml_installer = MedialibTestCasesInstaller(arch=arch, build_type=args.build_type,
                                            targets=args.targets,
                                            toolchain_dir_path=args.toolchain_dir_path,
                                            remote_machine_ip=args.remote_machine_ip,
                                            clean_build_dir=args.clean_build_dir,
                                            install_to_toolchain_rootfs=args.install_to_toolchain_rootfs,
                                            skip_ut=args.skip_ut, extra_flags=args.extra_flag)
    ml_installer.build()