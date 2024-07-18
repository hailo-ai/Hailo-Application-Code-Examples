################################################
# APP WRAPPER SOURCE
################################################
encoder_source = [
  'src/encoder/encoder.cpp'
  ]
detection_source = [
  'src/detection/yolov8.cpp',
  'src/detection/yolov8_postprocess.cpp'
  ]

json_dep = meson.get_compiler('c').find_library('libnlohmann_json_schema_validator', required: true, dirs: '/usr/lib/')

encoder_lib = shared_library('encoder',
  encoder_source,
  include_directories: include_directories('./include/'),
  dependencies : [gstreamer_deps, json_dep, gstmedialibrary_utils_dep, media_library_common_dep],
  version: meson.project_version(),
  gnu_symbol_visibility : 'default',
  install: true,
  install_dir: get_option('libdir'),
)


hailort_dep = [
meson.get_compiler('cpp').find_library('hailort', required: true)
]

xtensor_base_inc = include_directories('./include/media_library/open_source/xtensor_stack/base/', is_system: true)
xtensor_inc = [xtensor_base_inc]




detection_lib = shared_library('detection',
  detection_source,
  include_directories: [include_directories('./include/'), include_directories('./include/media_library'), include_directories('./include/media_library/open_source/xtensor_stack/base/'), include_directories('../media_library/include/')],
  dependencies : [hailort_dep, opencv_dep],
  version: meson.project_version(),
  gnu_symbol_visibility : 'default',
  install: true,
  install_dir: get_option('libdir'),
  c_args: '-std=gnu11',
)




opencv_dep = dependency('opencv4', version : '>= 4.0', method : 'pkg-config')


encoder_dep = declare_dependency(
  include_directories: [include_directories('./include/')],
  dependencies : [media_library_common_dep],
  link_with : encoder_lib)

detection_dep = declare_dependency(
  include_directories: [include_directories(['./include/'])],
  dependencies : [media_library_common_dep],
  link_with : detection_lib)


################################################
# VisionPreProc Example
################################################
vision_preproc_example_src = ['examples/vision_preproc_example.cpp']

executable('vision_preproc_example',
  vision_preproc_example_src,
  dependencies : gstreamer_deps + [media_library_frontend_dep, gstmedialibrary_utils_dep, encoder_dep, detection_dep],
  gnu_symbol_visibility : 'default',
  cpp_args: ['-I/usr/include/hailort/hailo', '-I/usr/include/opencv4/', '-O3', '-ffast-math'],
  install: true,
  install_dir: get_option('bindir'),
)

install_subdir('include/media_library', install_dir: get_option('includedir') + '/hailo')
