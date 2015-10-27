CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

PROJECT(deal.II-pod CXX)

FIND_PACKAGE(deal.II 8.4 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

INCLUDE_DIRECTORIES("include")

ADD_SUBDIRECTORY("source")
ADD_SUBDIRECTORY("programs")

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

ADD_CUSTOM_TARGET(run COMMAND ${CMAKE_SOURCE_DIR}/${TARGET}
  COMMENT "Run with ${CMAKE_BUILD_TYPE} configuration"
  )
