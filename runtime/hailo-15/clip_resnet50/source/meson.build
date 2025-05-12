project('gst-hailo-tools', 'c', 'cpp',
        version : '3.30.0',
        default_options : [ 'buildtype=release',
                            'c_std=c11', 'cpp_std=c++17']
       )

# Gstreamer Dependencies
gst_req = '>= 1.0.0'
gst_dep = dependency('gstreamer-1.0', version : gst_req,
  fallback : ['gstreamer', 'gst_dep'])
gst_base_dep = dependency('gstreamer-base-1.0', version : gst_req,
  fallback : ['gstreamer', 'gst_base_dep'])
gst_app_dep = dependency('gstreamer-app-1.0', version : gst_req,
  fallback : ['gstreamer', 'gst_app_dep'])
gstvideo_dep = dependency('gstreamer-video-1.0', version : gst_req,
  fallback : ['gst-plugins-base', 'video_dep'])

# Plugin Dependencies
gst_deps = [gst_dep, gst_base_dep, gstvideo_dep, gst_app_dep]

# Opencv Dependencies
opencv_dep = dependency('opencv4', version : '>= 4.0', method : 'pkg-config')

# ZMQ dep
zmq_dep = dependency('libzmq', method : 'pkg-config')

# Tracker Dep
tracker_dep = dependency('hailo_tracker', method : 'pkg-config')

# General Dep
tappas_general_dep = dependency('hailo_tappas_general', method : 'pkg-config')

# GST Image Handling
image_dep = dependency('tappas_image_utils', method : 'pkg-config')

libhailort_dep = [meson.get_compiler('cpp').find_library('hailort', required: false)]

# Library Arguments
hailo_lib_args = get_option('libargs')

# cxxopts Include Directories
cxxopts_inc_dir = get_option('libcxxopts')
cxxopts_inc = [include_directories(cxxopts_inc_dir, is_system: true)]

# Apps install dir
apps_install_dir = get_option('apps_install_dir')

# General Include Directories
hailo_general_inc = [include_directories('./general')]


################################################
# MEDIA LIBRARY DEPS
################################################
media_library_common_dep = dependency('hailo_media_library_common', method : 'pkg-config')
media_library_frontend_dep = dependency('hailo_media_library_frontend', method : 'pkg-config')
media_library_encoder_dep = dependency('hailo_media_library_encoder', method : 'pkg-config')
media_library_api_dep = dependency('hailo_media_library_api', method : 'pkg-config')
encoder_dep = dependency('hailo_encoder', method : 'pkg-config')
gstmedialibrary_utils_dep = dependency('gstmedialibutils', method : 'pkg-config')

dependencies_apps = gst_deps + media_library_encoder_dep + media_library_common_dep + media_library_frontend_dep + encoder_dep + media_library_api_dep + gstmedialibrary_utils_dep

# Hailort Arguments
sysroot_arg = []
if meson.is_cross_build()
    cmd_array = meson.get_compiler('cpp').cmd_array()
    foreach arg : cmd_array
        if arg.startswith('--sysroot=')
            sysroot = arg.split('=')[1]
            break
        endif
    endforeach
    message('Cross building using sysroot: ' + sysroot)
    if sysroot != ''
        sysroot_arg += '--sysroot=' + sysroot
    endif
endif


################################################
# HAILO 15 AI EXAMPLE APP
################################################

clip_example_app_src = ['clip_main.cpp']

executable('clip_example',
  clip_example_app_src,
  cpp_args : hailo_lib_args,
  include_directories:  cxxopts_inc,
  dependencies : dependencies_apps + [libhailort_dep, opencv_dep, tracker_dep, tappas_general_dep, image_dep, zmq_dep],
  gnu_symbol_visibility : 'default',
  install: true,
  install_dir: apps_install_dir + '/ai_example_app',
)

# install json config files
install_subdir('configs', install_dir : apps_install_dir + '/ai_example_app/resources')

tappas_workspace = './'

post_proc_install_dir = get_option('post_processes_install_dir')
if post_proc_install_dir == ''
  # By default install for x86 workspace
  post_proc_install_dir = apps_install_dir + '/ai_example_app'
endif

################################################
# CLIP SOURCES
################################################
clip_sources = [
    'postprocess/clip/clip.cpp',
]

shared_library('clip_post',
    clip_sources,
    cpp_args : hailo_lib_args,
    include_directories: [hailo_general_inc, include_directories('./')],
    dependencies :[tracker_dep, zmq_dep],
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: post_proc_install_dir,
)

if get_option('include_python')
endif