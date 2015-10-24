#include <algorithm>
#include <string>
#include <vector>
#include <glob.h>

#include <deal.II-rom/extra/extra.h>

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
}
