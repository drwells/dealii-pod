#include "extra.h"

namespace extra
{
  std::vector<std::string>
  expand_file_names
  (const std::string &file_name_glob)
  {
    std::vector<std::string> file_names;
    glob_t glob_result;
    glob(file_name_glob.c_str(), GLOB_TILDE, nullptr, &glob_result);
    for (unsigned int file_name_n = 0; file_name_n < glob_result.gl_pathc;
         ++file_name_n)
      {
        file_names.emplace_back(glob_result.gl_pathv[file_name_n]);
      }
    globfree(&glob_result);
    std::sort(file_names.begin(), file_names.end());
    return file_names;
  }


  std::string
  int_to_string
  (const unsigned int value, const unsigned int digits)
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
