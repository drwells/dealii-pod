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
#ifndef dealii__rom_compute_pod_parameters_h
#define dealii__rom_compute_pod_parameters_h
#include <deal.II/base/parameter_handler.h>

#include <string>

namespace POD
{
  using namespace dealii;

  class Parameters
  {
  public:
    int dimension;
    int fe_order;
    bool renumber;
    std::string snapshot_glob;
    std::string triangulation_file_name;

    int n_pod_vectors;
    bool center_trajectory;

    bool save_plot_pictures;

    void read_data(const std::string &file_name);
  private:
    void configure_parameter_handler(ParameterHandler &file_name) const;
  };
}
#endif
