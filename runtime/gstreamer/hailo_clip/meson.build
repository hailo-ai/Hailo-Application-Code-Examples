project('clip_app', 'c', 'cpp',
        version : '1.1.1',
        default_options : [ 'warning_level=1',
                            'buildtype=release',
                            'c_std=c11', 'cpp_std=c++17']
       )

postprocess_dep = dependency('hailo-tappas-core', version : '>=3.28.0', required : false)

if not postprocess_dep.found()
    postprocess_dep = dependency('hailo_tappas', version : '>=3.28.0', required : true)
endif
subdir('cpp')
