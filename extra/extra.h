#include <deal.II/base/types.h>
#include <deal.II/bundled/boost/lexical_cast.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <glob.h>

namespace extra {

  using namespace dealii;
  // This gets around the buggy `int_to_string` included with deal.II.
  std::string int_to_string
  (const unsigned int value,
   const unsigned int digits);

  std::vector<std::string>
  expand_file_names
  (const std::string &file_name_glob);
}
