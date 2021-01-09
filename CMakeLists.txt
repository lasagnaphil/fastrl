cmake_minimum_required(VERSION 3.17)
project(fastrl)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(MPI REQUIRED)
if (FASTRL_PYTHON)
    find_package(pybind11 REQUIRED)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(fastrl src/fastrl.cpp)
set_target_properties(fastrl PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_include_directories(fastrl PUBLIC include)

if (MPI_COMPILE_FLAGS)
    set_target_properties(fastrl PROPERTIES COMPILE_FLAGS ${MPI_COMPILE_FLAGS})
endif()

if (MPI_LINK_FLAGS)
    set_target_properties(fastrl PROPERTIES LINK_FLAGS ${MPI_LINK_FLAGS})
endif()

if (TARGET c10d)
    MESSAGE("c10d Found!")
endif()

set(FASTRL_LIBS ${TORCH_LIBRARIES} MPI::MPI_C)

add_subdirectory(deps/tensorboard_logger)
set(FASTRL_LIBS tensorboard_logger ${FASTRL_LIBS})

target_link_libraries(fastrl PUBLIC ${FASTRL_LIBS})

add_subdirectory(test)

if (FASTRL_PYTHON)
    add_subdirectory(python)
endif()
