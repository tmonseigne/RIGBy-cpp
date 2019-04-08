cmake_minimum_required (VERSION 3.12)

project (Riemann-Geometrie
		VERSION 1.0
		DESCRIPTION "Riemann-Geometrie"
		LANGUAGES CXX)

# CMAKE Parameters
set(CMAKE_CXX_STANDARD 11)		# Standard
set(CMAKE_BUILD_TYPE Release)	# Default Build
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/build")

# Dependencies List
FILE(GLOB children RELATIVE ${CMAKE_SOURCE_DIR}/dependencies/ ${CMAKE_SOURCE_DIR}/dependencies/*)
SET(DEPENDENCIES_LIST "")
FOREACH(child ${children})
	IF(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/dependencies/${child})
		LIST(APPEND DEPENDENCIES_LIST ${child})
	ENDIF()
ENDFOREACH()

message(STATUS "
\tProject Name : ${PROJECT_NAME} ${VERSION}
\tCmake Version : ${CMAKE_VERSION}
\tStandard : ${CMAKE_CXX_STANDARD}
\tBuild Type: ${CMAKE_BUILD_TYPE}
\tBuild Directory: ${CMAKE_INSTALL_PREFIX}
\tDependencies list : ${DEPENDENCIES_LIST}")

FILE(GLOB_RECURSE SRC_FILES src/* tests/*)
MESSAGE(SRC_FILES)

add_executable(GoogleTest SRC_FILES)

# Additional include directories
set_property(TARGET GoogleTest
  APPEND PROPERTY INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/dependencies/googletest/googletest
    ${CMAKE_SOURCE_DIR}/dependencies/googletest/googletest/include
    ${CMAKE_SOURCE_DIR}/dependencies/googletest/googlemock
    ${CMAKE_SOURCE_DIR}/dependencies/googletest/googlemock/include
    ${CMAKE_SOURCE_DIR}/dependencies/eigen
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src
)