SET(NS_ROM_TARGET_SRC
  ${NS_ROM_TARGET}.cc
  parameters.cc
  parameters.h
  rk_factory.h
  rk_factory.cc
  )

ADD_EXECUTABLE(${NS_ROM_TARGET} ${NS_ROM_TARGET_SRC})
DEAL_II_SETUP_TARGET(${NS_ROM_TARGET})
TARGET_LINK_LIBRARIES(${NS_ROM_TARGET} deal.II-pod)
