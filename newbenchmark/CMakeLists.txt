cmake_minimum_required(VERSION 3.13)
project(rotate_ben CXX)

find_package(HPX REQUIRED)
add_executable(rotate_ben rotate_ben.cpp)
target_link_libraries(rotate_ben HPX::hpx HPX::wrap_main)
