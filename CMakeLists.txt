cmake_minimum_required(VERSION 3.17)
project(fastrl)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(fastrl src/fastrl.cpp)
set_target_properties(fastrl PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_include_directories(fastrl PUBLIC include)

set(FASTRL_LIBS ${TORCH_LIBRARIES})

add_subdirectory(deps/tensorboard_logger)
set(FASTRL_LIBS tensorboard_logger ${FASTRL_LIBS})

target_link_libraries(fastrl PUBLIC ${FASTRL_LIBS})

add_subdirectory(test)
add_subdirectory(python)
