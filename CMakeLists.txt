# 避免不必要的警告
if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
if (MSVC)
  # VS 编译器
  add_compile_options(/W0)
else()
  # 其他
  add_compile_options(-w)
endif()
# CUDA支持需要CMake较高的版本
cmake_minimum_required(VERSION 3.10)
# 项目名称
project(PathTracer)
# 添加自定义CMake脚本
set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/CMake"
  ${CMAKE_MODULE_PATH}
  )
include(${CMAKE_SOURCE_DIR}/CMake/configure_optix.cmake)
# 设置属性
set_property ( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME} )
# 选取项目文件
aux_source_directory(src DIR_SRCS)
file(GLOB ImGui include/ImGui/*.h include/ImGui/*.cpp)
file(GLOB Shader shader/*.glsl)
file(GLOB HEADER_FILES src/*.h)
#add_executable(PathTracer ${DIR_SRCS} ${HEADER_FILES} ${ImGui} ${Shader})
source_group("ImGui" FILES ${ImGui})
source_group("Shader" FILES ${Shader})
file(COPY imgui.ini DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# 寻找依赖库
find_package(CUDA 11.0 REQUIRED)
find_package(OptiX REQUIRED)
# 包含目录
include_directories(include)
include_directories(include/glm)
include_directories(${OptiX_INCLUDE})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
# CUDA文件编译
cuda_compile_and_embed(embedded_ptx_code src/devicePrograms.cu)
# 向项目中添加文件
add_executable(PathTracer ${DIR_SRCS} ${HEADER_FILES} ${embedded_ptx_code}  ${ImGui} ${Shader})
# 设置链接库
if(CUDA_FOUND)
	target_link_libraries(PathTracer ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart_static.lib)
	target_link_libraries(PathTracer ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib)
endif()
target_link_libraries(PathTracer ${CMAKE_SOURCE_DIR}/lib/glfw3.lib)
target_link_libraries(PathTracer ${CMAKE_SOURCE_DIR}/lib/tinyobjloader.lib)
target_link_libraries(PathTracer opengl32.lib)
# 设置目标属性
set_target_properties(PathTracer PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(PathTracer PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(PathTracer PROPERTIES FOLDER "PathTracer")

