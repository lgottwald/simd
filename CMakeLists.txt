cmake_minimum_required(VERSION 2.8)

PROJECT(libsimd)

# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc
${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)
endif(DOXYGEN_FOUND)

# In my case the folder of includes can be any source folder,
# a practice very extended (in opposition to a single folder
# containing all the headers).
set ( libsimd_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} )

# Destination of the installed config files (relative path):
set ( CMAKE_CONFIG_DEST "share/cmake/libsimd" )

include ( CMakePackageConfigHelpers )

# We configure our template. The template is described later.
configure_package_config_file (                    
        "${CMAKE_CURRENT_SOURCE_DIR}/libsimd-config.cmake.in"
        # Important to write in CMAKE_BINARY_DIR if you want the registry
        # mechanism to work:
        "${CMAKE_BINARY_DIR}/libsimd-config.cmake"
        INSTALL_DESTINATION "${CMAKE_CONFIG_DEST}"
        PATH_VARS libsimd_INCLUDE_DIRS )

export ( PACKAGE libsimd )

# We redefine this variable, using this time a relative path:
set ( libsimd_INCLUDE_DIRS "include" )

# We write in the 'export' folder in order not to collide with
# the previous config file:
configure_package_config_file ( 
       "${CMAKE_CURRENT_SOURCE_DIR}/libsimd-config.cmake.in"
       "${CMAKE_BINARY_DIR}/export/libsimd-config.cmake"
       INSTALL_DESTINATION "${CMAKE_CONFIG_DEST}"
       PATH_VARS libsimd_INCLUDE_DIRS )


FILE(GLOB header_files "${CMAKE_CURRENT_SOURCE_DIR}/simd/*.hpp")
INSTALL(FILES ${header_files} DESTINATION include/simd)
INSTALL(FILES ${CMAKE_BINARY_DIR}/export/libsimd-config.cmake DESTINATION ${CMAKE_CONFIG_DEST})