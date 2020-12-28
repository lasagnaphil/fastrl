cmake_minimum_required(VERSION 3.17)
project(fastrl)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(fastrl src/fastrl.cpp)
target_include_directories(fastrl PUBLIC include)
target_link_libraries(fastrl "${TORCH_LIBRARIES}")
