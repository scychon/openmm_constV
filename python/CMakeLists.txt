#############################################
### Copy all source files to staging area ###
#############################################

# Create package directory structure
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mdtools)

file(GLOB STAGING_INPUT_FILES1 RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../mdtools/*.py"
)
foreach(file ${STAGING_INPUT_FILES1})
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${file}" "${CMAKE_CURRENT_BINARY_DIR}/mdtools/${file}" COPYONLY)
    set(STAGING_INPUT_FILES ${STAGING_INPUT_FILES} "${file}")
endforeach(file ${STAGING_INPUT_FILES1})

foreach(INIT_FILE ${STAGING_INPUT_FILES})
    set(infile "${CMAKE_CURRENT_SOURCE_DIR}/${INIT_FILE}")
    set(outfile "${CMAKE_CURRENT_BINARY_DIR}/${INIT_FILE}")
    add_custom_command(
        OUTPUT "${outfile}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${infile}" "${outfile}"
        DEPENDS "${infile}"
        COMMENT "CMake-copying file ${infile} to ${outfile}")
    set(STAGING_OUTPUT_FILES ${STAGING_OUTPUT_FILES} "${outfile}")
    # message("STAGING_OUTPUT_FILE = " ${outfile})
endforeach(INIT_FILE ${STAGING_INPUT_FILES})


set(WRAP_FILE ConstVPluginWrapper.cpp)
set(MODULE_NAME constvplugin)

# Execute SWIG to generate source code for the Python module.

add_custom_command(
    OUTPUT "${WRAP_FILE}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/constvplugin.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/constvplugin.i"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Compile the Python module.

add_custom_target(PythonInstall DEPENDS "${WRAP_FILE}")
set(CONSTVPLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(CONSTVPLUGIN_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
add_custom_command(TARGET PythonInstall
    COMMAND "${PYTHON_EXECUTABLE}" setup.py build
    COMMAND "${PYTHON_EXECUTABLE}" setup.py install
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

