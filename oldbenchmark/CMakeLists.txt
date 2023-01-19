cmake_minimum_required(VERSION 3.13)
project(hpx_rotate CXX)

find_package(HPX REQUIRED)
add_executable(hpx_rotate hpx_rotate.cpp)
target_link_libraries(hpx_rotate HPX::hpx HPX::wrap_main)
