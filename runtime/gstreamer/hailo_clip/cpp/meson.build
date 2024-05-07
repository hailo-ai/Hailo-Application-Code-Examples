
################################################
# CLIP SOURCES
################################################
clip_sources = [
    'clip.cpp',
]
shared_library('clip_post',
    clip_sources,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: join_paths(meson.project_source_root(), 'resources'),
)    
################################################
# fastsam SOURCES
################################################
fastsam_sources = [
    'fastsam_postprocess.cpp',
]
shared_library('fastsam_post',
    fastsam_sources,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: join_paths(meson.project_source_root(), 'resources'),
)    

################################################
# clip_cropper SOURCES
################################################
clip_croppers_sources = [
    'clip_croppers.cpp',
]
shared_library('clip_croppers',
    clip_croppers_sources,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: join_paths(meson.project_source_root(), 'resources'),
)    

################################################
# clip_matcher SOURCES
################################################

# sudo apt-get install libblas-dev liblapack-dev
# to find blas.pc
# find /usr -name '*blas*.pc'

cblas_dep = dependency('blas', dirs : '/usr/lib/x86_64-linux-gnu/pkgconfig')

clip_matcher_sources = [
    'clip_matcher.cpp','TextImageMatcher.cpp',
]
shared_library('clip_matcher',
    clip_matcher_sources,
   dependencies : [postprocess_dep, cblas_dep],
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: join_paths(meson.project_source_root(), 'resources'),
)    

################################################
# aspect_ratio_fix SOURCES
################################################
aspect_ratio_fix_sources = [
    'aspect_ratio_fix.cpp',
]
shared_library('aspect_ratio_fix',
    aspect_ratio_fix_sources,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: join_paths(meson.project_source_root(), 'resources'),
)  