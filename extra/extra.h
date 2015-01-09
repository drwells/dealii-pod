#include <iostream>
#include <string>

#include <deal.II/base/types.h>
#include <deal.II/bundled/boost/lexical_cast.hpp>

namespace extra {

  using namespace dealii;
  // This gets around the buggy `int_to_string` included with deal.II.
  std::string int_to_string
  (const unsigned int value,
   const unsigned int digits = numbers::invalid_unsigned_int);
}
