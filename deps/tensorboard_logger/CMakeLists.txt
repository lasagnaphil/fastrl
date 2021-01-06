cmake_minimum_required(VERSION 3.10)
project(tensorboard_logger)

set(CMAKE_CXX_STANDARD 17)

find_package(Protobuf REQUIRED)

# Note: Use the command ``make proto`` to generate protobuf C++ files before using this CMake script
file(GLOB TB_LOGGER_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
add_library(tensorboard_logger ${TB_LOGGER_SRC_FILES})
target_include_directories(tensorboard_logger PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tensorboard_logger PUBLIC protobuf::libprotobuf-lite)
