project('render', ['c', 'cpp'],
  default_options: [
    'prefix=' + meson.global_source_root() / 'packaged',
    'buildtype=debugoptimized',
    'warning_level=3',
    'werror=false',
    'cpp_std=c++20',
  ]
)

add_project_arguments(
  '-Wno-unused-parameter',
  '-Wno-documentation',
  language: ['c', 'cpp'],
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
  include_directories: [
    'src',
  ],
  install: true,
)

executable(
  'testy',
  'test.cc',
  dependencies: [
    dependency('gl'),
    dependency('glfw3'),
    dependency('glew'),
  ],
  include_directories: [
    'src',
  ],
  install: true,
)
