
aux_source_directory(. SRC)
add_library(cudart SHARED ${SRC})
target_link_libraries(cudart cuda)

install(TARGETS cudart DESTINATION lib)
install(FILES cuda_runtime_api.h    DESTINATION include)
install(FILES channel_descriptor.h  DESTINATION include)
