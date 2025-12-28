
include_directories(${CMAKE_CURRENT_LIST_DIR}/../src/cudart/)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../src/cuda/)

link_libraries(cuda cudart cufft)

add_executable(cuda_device    cuda_device.cc)
add_executable(cuda_memory    cuda_memory.cc)
add_executable(cuda_execute   cuda_execute.cc)
