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
\tBuild Directo${PROJECT_NAME}: ${CMAKE_INSTALL_PREFIX}
\tDependencies ${PROJECT_NAME}st : ${DEPENDENCIES_LIST}")

# Basic sources
FILE(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/* ${CMAKE_SOURCE_DIR}/tests/*)

#Google tests

add_definitions(-DGTEST_LANGUAGE_CXX11)

find_package(Threads REQUIRED)

if ($ENV{GOOGLETEST_DIR})
	SET(GOOGLETEST_DIR $ENV{GOOGLETEST_DIR})
else ()
	message(WARNING "Using googletest src dir specified at Qt Creator wizard")
	SET(GOOGLETEST_DIR "${CMAKE_SOURCE_DIR}/dependencies/googletest")
endif ()
if (EXISTS ${GOOGLETEST_DIR})
	SET(GTestSrc ${GOOGLETEST_DIR}/googletest)
	SET(GMockSrc ${GOOGLETEST_DIR}/googlemock)
else ()
	message( FATAL_ERROR "No googletest src dir found - set GOOGLETEST_DIR to enable!")
endif ()

#executable
add_executable(${PROJECT_NAME}	
				${SRC_FILES} 
				${GTestSrc}/src/gtest-all.cc
				${GMockSrc}/src/gmock-all.cc
			  )

# Additional include directories
set_property(TARGET ${PROJECT_NAME}
	APPEND PROPERTY INCLUDE_DIRECTORIES
		${CMAKE_SOURCE_DIR}/dependencies/googletest/googletest
		${CMAKE_SOURCE_DIR}/dependencies/googletest/googletest/include
		${CMAKE_SOURCE_DIR}/dependencies/googletest/googlemock
		${CMAKE_SOURCE_DIR}/dependencies/googletest/googlemock/include
		${CMAKE_SOURCE_DIR}/dependencies/eigen
		${CMAKE_SOURCE_DIR}/tests
		${CMAKE_SOURCE_DIR}/src
		${GTestSrc} 
		${GTestSrc}/include 
		${GMockSrc} 
		${GMockSrc}/include
)

add_test(${PROJECT_NAME} COMMAND ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)