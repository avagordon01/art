project('ade', 'cpp',
  default_options: [
    'prefix=' + meson.source_root() / 'packaged',
    'buildtype=debugoptimized',
    'warning_level=3',
    'werror=true',
    'cpp_std=c++17',
  ]
)

add_project_arguments(
  '-Wno-unused-parameter',
  '-Wno-missing-field-initializers',
  language: 'cpp',
)

cc = meson.get_compiler('cpp')
m_dep = cc.find_library('m', required : false)
gl_dep = dependency('gl')
glfw_dep = dependency('glfw3')
glm_dep = dependency('glm')
freetype_dep = dependency('freetype2')

executable('render', 'src/render.cc', dependencies: [m_dep, gl_dep, glfw_dep, glm_dep, freetype_dep], install: true)
