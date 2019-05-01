# 设置cmake的最小版本
cmake_minimum_required(VERSION 3.10)
# 项目名称
project(demo C CXX)
#设置c++编译器
set(CMAKE_CXX_STANDARD 11)

# 设置TENSORFLOW_DIR变量，变量内容为安装的tensorflow文件夹路径
set(TENSORFLOW_DIR $ENV{HOME}/tensorflow)
#项目中的include路径
include_directories(${TENSORFLOW_DIR})
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/proto)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/protobuf-host/include)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/protobuf/src)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/absl)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/eigen)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/nsync/public)
include_directories(${TENSORFLOW_DIR}/bazel-genfiles)

# 项目中lib路径
link_directories(${TENSORFLOW_DIR}/bazel-bin/tensorflow)
link_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/lib)
link_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/protobuf-host/lib)
#link_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11)

add_executable(demo main.cpp)

# 连接libtensorflow_cc.so和libtensorflow_framework库。
#target_link_libraries(demo tensorflow_framework tensorflow_cc)
target_link_libraries(demo
    ${TENSORFLOW_DIR}/bazel-bin/tensorflow/libtensorflow_framework.so.1
    ${TENSORFLOW_DIR}/bazel-bin/tensorflow/libtensorflow_cc.so)
