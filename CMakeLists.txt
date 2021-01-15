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
target_include_directories(fastrl PUBLIC include ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

if (MPI_COMPILE_FLAGS)
    set_target_properties(fastrl PROPERTIES COMPILE_FLAGS ${MPI_COMPILE_FLAGS})
endif()

if (MPI_LINK_FLAGS)
    set_target_properties(fastrl PROPERTIES LINK_FLAGS ${MPI_LINK_FLAGS})
endif()

if (FASTRL_MPI)
    target_compile_definitions(fastrl PUBLIC FASTRL_MPI=1)
endif()
if (FASTRL_GLOO)
    target_compile_definitions(fastrl PUBLIC FASTRL_GLOO=1)
endif()
if (FASTRL_NCCL)
    target_compile_definitions(fastrl PUBLIC FASTRL_NCCL=1)
endif()

add_subdirectory(deps/tensorboard_logger)
add_subdirectory(deps/toml11)

set(FASTRL_LIBS ${TORCH_LIBRARIES} ${MPI_LIBRARIES}
        ${TORCH_PREFIX_PATH}/lib/libc10d.a ${TORCH_PREFIX_PATH}/lib/libgloo.a
        pthread toml11)

set(FASTRL_LIBS tensorboard_logger ${FASTRL_LIBS})

target_link_libraries(fastrl PUBLIC ${FASTRL_LIBS})

add_subdirectory(test)

if (FASTRL_PYTHON)
    add_subdirectory(python)
endif()
