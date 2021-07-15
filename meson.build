project('render', 'cpp',
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
  language: 'cpp',
)

executable(
  'render',
  'src/render.cc',
  dependencies: [
    dependency('gl'),
    dependency('glfw3'),
    dependency('glm'),
    dependency('freetype2'),
    dependency('assimp'),
  ],
  install: true,
)
