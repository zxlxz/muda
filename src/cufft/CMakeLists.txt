
aux_source_directory(. SRC)
add_library(cufft ${SRC})
target_link_libraries(cufft "-framework Accelerate")
install(FILES cufft.h DESTINATION include)
