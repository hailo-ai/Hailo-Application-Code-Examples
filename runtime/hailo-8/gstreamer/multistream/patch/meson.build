# Project Configuration
core_conf = configuration_data()
core_conf.set('PACKAGE', '"@0@"'.format(meson.project_name()))
core_conf.set('VERSION', '"@0@"'.format(meson.project_version()))
configure_file(output : 'config.h', configuration : core_conf)
common_args = ['-DHAVE_CONFIG_H']
sysroot_arg = []
if meson.is_cross_build()
    meson_version = meson.version()
    if meson_version.version_compare('>=0.54.0')
        sysroot = meson.get_external_property('sys_root', '')
    else
        sysroot = meson.get_cross_property('sys_root', '')
    endif
    if sysroot != ''
        sysroot_arg += '--sysroot=' + sysroot
    endif
endif
hailo_link_args = []

plugin_sources = [
    'gsthailotools.cpp',
    'filter/gsthailofilter.cpp',
    'filter/gsthailocounter.cpp',
    'muxer/gsthailomuxer.cpp',
    'muxer/gsthailoroundrobin.cpp',
    'muxer/gsthailostreamrouter.cpp',
    'common/image.cpp',
    'overlay/overlay.cpp',
    'overlay/gsthailooverlay.cpp',
    'cropping/gsthailobasecropper.cpp',
    'cropping/gsthailocropper.cpp',
    'cropping/gsthailoaggregator.cpp',
    'tiling/gsthailotilecropper.cpp',
    'tiling/gsthailotileaggregator.cpp',
    'tracking/gsthailotracker.cpp',
    'gallery/gsthailogallery.cpp',
    'export/export_file/gsthailoexportfile.cpp',
    'export/export_zmq/gsthailoexportzmq.cpp',
    'import/import_zmq/gsthailoimportzmq.cpp',
]

# equivalent to - dl_dep = dependency('dl')
dl_dep = meson.get_compiler('c').find_library('dl', required : false)

# ZMQ dep
zmq_dep = dependency('libzmq', method : 'pkg-config')
gsthailotools_deps = plugin_deps + [meta_dep, dl_dep, opencv_dep, tracker_dep, zmq_dep]

################################################
# Hailo Gstreamer Shared Library
################################################
shared_library('gsthailotools',
    plugin_sources,
    cpp_args : hailo_lib_args + common_args + sysroot_arg+['-pthread']+['-g'],
    c_args : hailo_lib_args + common_args + sysroot_arg,
    link_args: hailo_link_args,
    include_directories: [hailo_general_inc, xtensor_inc, rapidjson_inc, hailo_mat_inc],
    dependencies : gsthailotools_deps + [dependency('threads')],
    gnu_symbol_visibility : 'default',
    version: meson.project_version(),
    install: true,
    install_dir: get_option('libdir') + '/gstreamer-1.0/',
)

if not get_option('include_python')
    subdir_done()
endif

fs_dep = meson.get_compiler('c').find_library('stdc++fs', required : true)

################################################
# Hailo Python Gstreamer Shared Library
################################################
python_sources = [
    'python/gsthailopython.cpp',
    'python/hailopython_infra.cpp',
]

shared_library('gsthailopython',
    python_sources,
    cpp_args : hailo_lib_args + common_args,
    include_directories: [hailo_general_inc, xtensor_inc],
    dependencies : plugin_deps + [fs_dep, python_dep, meta_dep, dl_dep] + gx_deps,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: get_option('libdir') + '/gstreamer-1.0/',
)

################################################
# Hailo Python Module
################################################

hailo_python_sources = [
    'python/hailo_python_api.cpp'
]

python_installation.extension_module('hailo',
    hailo_python_sources,
    include_directories: hailo_general_inc,
    dependencies : [pybind_dep],
    link_language : 'cpp',
    override_options: ['cpp_rtti=true'],
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: python_package_install_dir,
)
