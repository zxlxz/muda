
function(add_metallib _src_name)
    get_filename_component(_src_path  ${_src_name} ABSOLUTE)
    get_filename_component(_base_name ${_src_name} NAME_WE)
    set(_out_name "${_base_name}.metallib")
    set(_out_path ${CMAKE_CURRENT_BINARY_DIR}/${_out_name})
    message("${_src_path} => ${_out_path}")

    add_custom_command(
        OUTPUT "${_out_path}"
        COMMAND xcrun -sdk macosx metal "${_src_path}" -o "${_out_path}"
        DEPENDS "${_src_path}"
        COMMENT "Compiling Metal: ${_src_path}"
        VERBATIM
    )

    add_custom_target("${_base_name}"
      DEPENDS "${_out_path}"
      WORKING_DIRECTORY ${src_path}
    )
endfunction()
