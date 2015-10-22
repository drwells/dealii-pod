/* ---------------------------------------------------------------------
 * Copyright (C) 2014 David Wells
 *
 * This file is NOT part of the deal.II library.
 *
 * This file is free software; you can use it, redistribute it, and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 * This program is based on step-26 of the deal.ii library.
 *
 * Author: David Wells, Virginia Tech, 2014
 */
#ifndef dealii__rom_rom_error_h
#define dealii__rom_rom_error_h
#include <string>

class Parameters
{
public:
  std::string snapshot_glob;
  std::string pod_vector_glob;
  std::string mean_vector_file_name;
  std::string pod_coefficients_file_name;
  bool renumber;
  unsigned int fe_order;

  double snapshot_start_time;
  double snapshot_stop_time;

  double rom_start_time;
  double rom_stop_time;
  Parameters();
};

#endif
