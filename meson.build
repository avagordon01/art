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
  language: 'cpp',
)

common_deps = [
  dependency('glfw3'),
  dependency('glm'),
  dependency('freetype2'),
  dependency('assimp'),
  dependency('range-v3'),
]

executable(
  'render',
  'src/render.cc',
  dependencies: [
    dependency('gl'),
  ] + common_deps,
  include_directories: [
    'src',
  ],
  install: true,
)
