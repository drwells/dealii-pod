/* ---------------------------------------------------------------------
 * Copyright (C) 2014-2015 David Wells
 *
 * This file is NOT part of the deal.II library.
 *
 * This file is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 *
 * Author: David Wells, Virginia Tech, 2014-2015;
 *         David Wells, Rensselaer Polytechnic Institute, 2015
 */
#include <string>
#include <vector>

namespace extra
{
  std::vector<std::string>
  expand_file_names
  (const std::string &file_name_glob);
}
