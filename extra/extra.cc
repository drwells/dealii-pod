#include "extra.h"

namespace extra
{
  std::string
  int_to_string (const unsigned int value, const unsigned int digits)
  {
    std::string lc_string = boost::lexical_cast<std::string>(value);

    if (digits == numbers::invalid_unsigned_int)
      return lc_string;
    else if (lc_string.size() < digits)
      {
        std::string padding(digits - lc_string.size(), '0');
        lc_string.insert(0, padding);
      }
    return lc_string;
  }
}
