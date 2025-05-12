################################################
# example_app
################################################
example_app_sources = [
'example_app.cpp', 'hailo_app_data_aggregator.cpp', 'hailo_app_useful_funcs.cpp',
]

executable('example_app',
    example_app_sources,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: meson.project_source_root(),
)    
