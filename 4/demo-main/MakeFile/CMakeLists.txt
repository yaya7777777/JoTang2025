# 指定CMake的最低版本要求
cmake_minimum_required(VERSION 3.10)

# 定义项目名称和版本
project(demo VERSION 1.0)

# 指定C标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

# 添加可执行文件并指定源文件
add_executable(demo.out hello.c lib/add.c lib/mu.c)

# 设置编译选项
target_compile_options(demo.out PRIVATE 
    -g 
    -O2 
    -Wall 
    -Werror
)

# 指定头文件包含路径
target_include_directories(demo.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加自定义清理目标（不能使用clean，这是CMake保留名称）
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove -f demo.out
    COMMAND ${CMAKE_COMMAND} -E remove_directory CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove -f CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove -f cmake_install.cmake
    COMMAND ${CMAKE_COMMAND} -E remove -f Makefile
    COMMENT "Cleaning up all build files"
)