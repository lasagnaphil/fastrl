find_package(raylib 3.0.0 REQUIRED)

add_executable(test_pendulum_env test_pendulum_env.cpp)
target_link_libraries(test_pendulum_env PRIVATE fastrl raylib)

add_executable(test_mountaincar_env test_mountain_car_env.cpp)
target_link_libraries(test_mountaincar_env PRIVATE fastrl raylib)
