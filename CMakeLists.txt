cmake_minimum_required (VERSION 3.12)

include(GoogleTest)
include(ProcessorCount)

project (Riemann-Geometrie
		VERSION 1.0
		DESCRIPTION "Riemann-Geometrie"
		LANGUAGES CXX)

# CMAKE Parameters
set(CMAKE_CXX_STANDARD 11)		# Standard
set(CMAKE_BUILD_TYPE Release)	# Default Build
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/build")

# multiproc
ProcessorCount(N)
if(NOT N EQUAL 0)
  set(CTEST_BUILD_FLAGS -j${N})
  set(ctest_test_args ${ctest_test_args} PARALLEL_LEVEL ${N})
endif()

# Dependencies List
file(GLOB children RELATIVE ${CMAKE_SOURCE_DIR}/dependencies/ ${CMAKE_SOURCE_DIR}/dependencies/*)
set(DEPENDENCIES_LIST "")
foreach(child ${children})
	if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/dependencies/${child})
		list(APPEND DEPENDENCIES_LIST ${child})
	endif()
endforeach()

# Basic sources
file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/* ${CMAKE_SOURCE_DIR}/include/* ${CMAKE_SOURCE_DIR}/test/*)

#################################################
########## Code Coverage Configuration ##########
#################################################
add_library(coverage_config INTERFACE)

option(CODE_COVERAGE "Enable coverage reporting" OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	# Add required flags (GCC & LLVM/Clang)
	target_compile_options(coverage_config INTERFACE
		-O0        # no optimization
		-g         # generate debug info
		--coverage # sets all required flags
	)
	if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
		target_link_options(coverage_config INTERFACE --coverage)
	else()
		target_link_libraries(coverage_config INTERFACE --coverage)
	endif()
endif()

################################################
########## Google tests Configuration ##########
################################################
add_definitions(-DGTEST_LANGUAGE_CXX11)

find_package(Threads REQUIRED)

if($ENV{GOOGLETEST_DIR})
	set(GOOGLETEST_DIR $ENV{GOOGLETEST_DIR})
else ()
	set(GOOGLETEST_DIR "${CMAKE_SOURCE_DIR}/dependencies/googletest")
endif ()
if(EXISTS ${GOOGLETEST_DIR})
	set(GTestSrc ${GOOGLETEST_DIR}/googletest)
	set(GMockSrc ${GOOGLETEST_DIR}/googlemock)
else ()
	message( FATAL_ERROR "No googletest src dir found - set GOOGLETEST_DIR to enable!")
endif()

####################################
########## Message Status ##########
####################################
message(STATUS "
\tProject Name : ${PROJECT_NAME} ${VERSION}
\tProcessor Count : ${N}
\tCmake Version : ${CMAKE_VERSION}
\tStandard : ${CMAKE_CXX_STANDARD}
\tCommpiler : ${CMAKE_CXX_COMPILER}
\tBuild Type: ${CMAKE_BUILD_TYPE}
\tBuild Directoies : ${CMAKE_INSTALL_PREFIX}
\tDependencies : ${DEPENDENCIES_LIST}
\tCode coverage : ${CODE_COVERAGE}")

################################
########## Executable ##########
################################
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
		${CMAKE_SOURCE_DIR}/test
		${CMAKE_SOURCE_DIR}/include
		${GTestSrc} 
		${GTestSrc}/include 
		${GMockSrc} 
		${GMockSrc}/include
)

# link
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)
target_link_libraries(${PROJECT_NAME} PUBLIC coverage_config)

enable_testing()
gtest_discover_tests(${PROJECT_NAME})