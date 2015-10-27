SET(TARGET "compute-pod-matrices")

SET(TARGET_SRC
  ${TARGET}.cc
  parameters.cc
  parameters.h
  )

ADD_EXECUTABLE(${TARGET} ${TARGET_SRC})
DEAL_II_SETUP_TARGET(${TARGET})
TARGET_LINK_LIBRARIES(${TARGET} deal.II-pod)
