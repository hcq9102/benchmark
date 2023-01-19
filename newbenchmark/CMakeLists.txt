cmake_minimum_required(VERSION 3.13)
project(rotate_async CXX)

find_package(HPX REQUIRED)
add_executable(rotate_async rotate_async.cpp)
target_link_libraries(rotate_async HPX::hpx HPX::wrap_main)
