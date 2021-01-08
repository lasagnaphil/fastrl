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

# set(TBLOGGER_SRC_FILES src/crc.cc src/tensorboard_logger.cc ${TBLOGGER_PROTO_SRCS} ${TBLOGGER_PROTO_HDRS})
file(GLOB TBLOGGER_SRC_FILES src/*.cc)
add_library(tensorboard_logger ${TBLOGGER_SRC_FILES})
target_include_directories(tensorboard_logger PUBLIC include)
target_link_libraries(tensorboard_logger PRIVATE protobuf::libprotobuf)
