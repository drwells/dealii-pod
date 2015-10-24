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
#ifndef dealii__rom_ns_parameters_h
#define dealii__rom_ns_parameters_h
#include <deal.II/base/parameter_handler.h>

#include <fstream>

using namespace dealii;
namespace POD
{
  enum class FilterModel
    {
      Differential,
      L2Projection,
      PostDifferentialFilter,
      PostL2ProjectionFilter,
      LerayHybrid,
      ADLavrentiev,
      ADTikonov
    };

  namespace NavierStokes
  {
    class Parameters
    {
    public:
      int outflow_label;
      double reynolds_n;
      bool renumber;
      int fe_order;

      POD::FilterModel filter_model;
      double noise_multiplier;
      double lavrentiev_parameter;
      double filter_radius;
      unsigned int cutoff_n;
      bool filter_mean;

      double initial_time;
      double final_time;
      double time_step;

      int patch_refinement;
      double output_plot_time_start;
      double output_plot_time_stop;
      int output_interval;
      bool save_plot_pictures;

      void read_data(const std::string &file_name);
    private:
      void configure_parameter_handler(ParameterHandler &parameter_handler) const;
    };
  }
}
#endif
