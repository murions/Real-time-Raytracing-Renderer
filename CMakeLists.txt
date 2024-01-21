# 避免不必要的警告
if (POLICY CMP0048)
	cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
if (MSVC) # VS 编译器
	add_compile_options(/W0)
else() # 其他
	add_compile_options(-w)
endif()
# CUDA支持需要CMake较高的版本
cmake_minimum_required(VERSION 3.15)
# 项目名称
project(PathTracer)
# VS初始目录
set_property ( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME} )
# 添加自定义cmake脚本
set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/cmake"
  ${CMAKE_MODULE_PATH}
  )
include(${CMAKE_SOURCE_DIR}/CMake/configure_optix.cmake)
# 设置代码路径宏
add_compile_definitions(PROJECT_DIR="${CMAKE_SOURCE_DIR}")
# 选取项目文件
file(GLOB srcs ${CMAKE_SOURCE_DIR}/src/*.h ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/include/glad/glad.c)
file(GLOB ImGui ${CMAKE_SOURCE_DIR}/include/ImGui/*.h ${CMAKE_SOURCE_DIR}/include/ImGui/*.cpp)
file(GLOB Shader ${CMAKE_SOURCE_DIR}/shader/*.glsl)
file(GLOB HEADER_FILES ${CMAKE_SOURCE_DIR}/src/*.h)
# VS分组
source_group("ImGui" FILES ${ImGui})
source_group("Shader" FILES ${Shader})
file(COPY ${CMAKE_SOURCE_DIR}/imgui.ini DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# 寻找依赖库
find_package(CUDA 11.0 REQUIRED)
# 包含目录
link_directories(${CMAKE_SOURCE_DIR}/lib)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/glm)
include_directories(${CMAKE_SOURCE_DIR}/include/optix)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
# CUDA文件编译
option(COMPILE_CUDA_SHADERS "option for compiling .cu files" OFF)
if(COMPILE_CUDA_SHADERS)
	cuda_compile_and_embed(embedded_ptx_code src/devicePrograms.cu)
endif()
# 向项目中添加文件
if(COMPILE_CUDA_SHADERS)
	add_executable(${PROJECT_NAME} ${srcs} ${HEADER_FILES} ${embedded_ptx_code}  ${ImGui} ${Shader})
else()
	file(GLOB ptx_shader ${CMAKE_SOURCE_DIR}/shader/shaders.c)
	add_executable(${PROJECT_NAME} ${srcs} ${HEADER_FILES} ${ptx_shader} ${ImGui} ${Shader})
endif()
# 设置链接库
if(CUDA_FOUND)
	target_link_libraries(${PROJECT_NAME} ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart_static.lib)
	target_link_libraries(${PROJECT_NAME} ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib)
endif()
target_link_libraries(${PROJECT_NAME} glfw3.lib)
# 更新shader文件
if(COMPILE_CUDA_SHADERS)
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy
		${embedded_ptx_code} ${CMAKE_SOURCE_DIR}/shader/shaders.c
	)
endif()
# 设置目标属性
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "${PROJECT_NAME}")

