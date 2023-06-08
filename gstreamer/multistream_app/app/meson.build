################################################
# multistream_app
################################################
thread_dep = dependency('threads')

multistream_app_sources = [
'multistream_app.cpp', 'SrcBin.cpp', 'hailo_app_useful_funcs.cpp',
]


executable('multistream_app',
    multistream_app_sources,
    dependencies : [postprocess_dep, thread_dep],
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: meson.project_source_root(),
) 
