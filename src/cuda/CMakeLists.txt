aux_source_directory(. SRC)
add_library(cuda SHARED ${SRC})

target_link_libraries(cuda "-framework Metal" "-framework Foundation")

install(TARGETS cuda DESTINATION lib)
install(FILES cuda.h DESTINATION include)
