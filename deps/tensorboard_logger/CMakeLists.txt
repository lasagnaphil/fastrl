cmake_minimum_required(VERSION 3.10)
project(tensorboard_logger)

set(CMAKE_CXX_STANDARD 17)

# protobuf_generate_cpp(TBLOGGER_PROTO_SRCS TBLOGGER_PROTO_HDRS
#         event.proto
#         projector_config.proto
#         resource_handle.proto
#         summary.proto
#         tensor.proto
#         tensor_shape.proto
#         types.proto)

set(TBLOGGER_SRC_FILES src/crc.cc src/tensorboard_logger.cc)
set(TBLOGGER_HDR_FILES src/crc.h src/tensorboard_logger.h)
set(TBLOGGER_PROTO_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/event.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/projector_config.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/resource_handle.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/summary.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/tensor.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/tensor_shape.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/types.proto)
set(TBLOGGER_PROTO_SRC_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/event.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/projector_config.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/resource_handle.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/summary.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor_shape.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/types.pb.cc)
set(TBLOGGER_PROTO_HDR_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/event.pb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/projector_config.pb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/resource_handle.pb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/summary.pb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tensor.pb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tensor_shape.pb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/types.pb.h)

add_custom_command(
        OUTPUT ${TBLOGGER_PROTO_HDR_FILES} ${TBLOGGER_PROTO_SRC_FILES}
        COMMAND make proto
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS ${TBLOGGER_PROTO_FILES}
)
add_custom_target(compile_protobuf DEPENDS ${TBLOGGER_PROTO_HDR_FILES} ${TBLOGGER_PROTO_SRC_FILES})

add_library(tensorboard_logger ${TBLOGGER_SRC_FILES} ${TBLOGGER_PROTO_SRC_FILES})
target_include_directories(tensorboard_logger PUBLIC include)
target_link_libraries(tensorboard_logger PRIVATE protobuf::libprotobuf)
add_dependencies(tensorboard_logger compile_protobuf)