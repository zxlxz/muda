
aux_source_directory(. SRC)
add_library(cudart ${SRC})

install(TARGETS cudart DESTINATION lib)
install(FILES cuda_runtime_api.h DESTINATION include)
