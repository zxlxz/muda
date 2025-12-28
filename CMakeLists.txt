cmake_minimum_required(VERSION 3.20)

project(muda)

set(CMAKE_CXX_STANDARD 23)
include_directories(${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(src)
add_subdirectory(test)
