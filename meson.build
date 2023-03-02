project('render', 'cpp',
  default_options: [
    'prefix=' + meson.source_root() / 'packaged',
    'buildtype=debugoptimized',
    'warning_level=3',
    'werror=true',
    'cpp_std=c++20',
  ]
)

add_project_arguments(
  '-Wno-unused-parameter',
  '-Wno-documentation',
  language: 'cpp',
)

cmake = import('cmake')
units = cmake.subproject('units')
units_lib = units.dependency('units')

executable(
  'render',
  'src/render.cc',
  dependencies: [
    dependency('gl'),
    dependency('glfw3'),
    dependency('glm'),
    dependency('freetype2'),
    dependency('assimp'),
    dependency('range-v3'),
    units_lib,
  ],
  cpp_args: [
    '-Wno-documentation',
  ],
  include_directories: [
    'src',
  ],
  install: true,
)
