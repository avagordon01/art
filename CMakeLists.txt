set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
  CACHE STRING "Vcpkg toolchain file")
cmake_minimum_required(VERSION 3.21)
project(render)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-error -Wno-unused-parameter -Wno-documentation)

find_package(llnl-units CONFIG REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Freetype REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(range-v3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(pugixml CONFIG REQUIRED)

add_executable(render src/render.cc)
target_include_directories(render PRIVATE src)
target_link_libraries(render PRIVATE llnl-units::units OpenGL::OpenGL glfw glm::glm Freetype::Freetype assimp::assimp range-v3 nlohmann_json pugixml)
install(TARGETS render)
