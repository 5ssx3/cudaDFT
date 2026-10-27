cmake_minimum_required(VERSION 3.20)

# 在 cudaDFT 根目录执行以下命令：
# 配置：RTX 5090 使用架构 120；其他 GPU 请修改 CMAKE_CUDA_ARCHITECTURES。
# env -u CPATH -u CPLUS_INCLUDE_PATH cmake -S . -B build \
#   -DCMAKE_BUILD_TYPE=Release \
#   -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
#   -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
#   -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
#   -DCMAKE_PREFIX_PATH="$HOME/.local" \
#   -DCMAKE_CUDA_ARCHITECTURES=120
# 编译：中间产物保存在 build，可执行文件为 bin/cudaDFT。
# env -u CPATH -u CPLUS_INCLUDE_PATH cmake --build build --parallel
# 运行：将 /path/to/input.yaml 替换为实际输入文件。
# ./bin/cudaDFT /path/to/input.yaml
# 清除 CPATH/CPLUS_INCLUDE_PATH 是为了避免 Intel 环境头文件干扰 CUDA 编译。

set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures (RTX 5090: 120)")
project(cudaDFT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

# iks.cpp 包含 CUDA/Thrust 代码，按 CUDA 源文件编译。
set_source_files_properties(src/iks.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cudaDFT
    src/iks.cpp
    src/cuda_math.cu
    src/Davidson.cpp
    src/LBFGS.cpp
    src/read_input.cpp
    src/structure.cpp
    src/write.cpp
    src/smearing.cpp
)

target_include_directories(cudaDFT PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(cudaDFT PRIVATE
    yaml-cpp::yaml-cpp
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::curand
    CUDA::cufft
)

set_target_properties(cudaDFT PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    CUDA_RUNTIME_LIBRARY Shared
)
