SET(TARGET "rom")

SET(TARGET_SRC
  ${TARGET}.cc
  ${TARGET}.h
  parameters.cc
  parameters.h
  rk_factory.h
  rk_factory.cc
  )

ADD_EXECUTABLE(${TARGET} ${TARGET_SRC})
DEAL_II_SETUP_TARGET(${TARGET})
TARGET_LINK_LIBRARIES(${TARGET} deal.II-pod)
