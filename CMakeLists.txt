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


add_subdirectory("src")
add_subdirectory("tests")
