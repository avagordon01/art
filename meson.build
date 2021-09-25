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

vulkan_deps = [
  meson.get_compiler('cpp').find_library('dl', required: false),
  declare_dependency(
    link_args: ['-lglslang'],
  ),
  declare_dependency(
    link_args: ['-lSPIRV'],
  ),
  dependency('vulkan'),
  dependency('xcb'),
]

executable(
  'vulkan',
  'src/vulkan/vulkan.cc',
  'subprojects/utils/math.cpp',
  'subprojects/utils/shaders.cpp',
  'subprojects/utils/utils.cpp',
  include_directories: [
    'subprojects/raii_utils',
    'subprojects/utils',
  ],
  cpp_args: [
    '-DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1',
  ],
  dependencies: common_deps + vulkan_deps,
  install: true,
)
