/* ---------------------------------------------------------------------
 * $Id: project.cc $
 *
 * Copyright (C) 2015 David Wells
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
 *
 * Author: David Wells, Virginia Tech, 2015
 */
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/bundled/boost/archive/text_iarchive.hpp>
// needed to get around the "save the dof handler issue"
#include <deal.II/dofs/dof_faces.h>
#include <deal.II/dofs/dof_levels.h>

#include <algorithm>
#include <iostream>
#include <math.h>
#include <vector>

#include "../h5/h5.h"
#include "../pod/pod.h"
#include "../extra/extra.h"
#include "parameters.h"

constexpr int dim {3};

using namespace dealii;
int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Parameters parameters;
  parameters.read_data("parameter-file.prm");

  const FE_Q<dim> fe(parameters.fe_order);
  Triangulation<dim> triangulation;
  DoFHandler<dim> dof_handler;

  POD::create_dof_handler_from_triangulation_file
    (parameters.triangulation_file_name, parameters.renumber, fe, dof_handler,
     triangulation);

  auto file_names = extra::expand_file_names(parameters.snapshot_glob);

  std::vector<XDMFEntry> xdmf_entries;
  for (unsigned int snapshot_n = 0; snapshot_n < file_names.size(); ++snapshot_n)
    {
      auto &file_name = file_names.at(snapshot_n);
      BlockVector<double> snapshot;
      H5::load_block_vector(file_name, snapshot);
      auto &y_block = snapshot.block(1);

      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      std::vector<std::string> solution_name(1, "v");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        component_interpretation(1, DataComponentInterpretation::component_is_scalar);
      data_out.add_data_vector(y_block, solution_name, DataOut<dim>::type_dof_data,
                               component_interpretation);
      data_out.build_patches(parameters.patch_level);
      bool save_mesh = false;
      if (snapshot_n == 0)
        {
          save_mesh = true;
        }
      std::string mesh_file_name = "mesh.h5";
      std::string solution_file_name = "y-velocity-"
        + extra::int_to_string(snapshot_n, 10)
        + ".h5";
      std::string xdmf_filename = "solution.xdmf";
      DataOutBase::DataOutFilter data_filter
        (DataOutBase::DataOutFilterFlags(true, true));
      data_out.write_filtered_data(data_filter);
      data_out.write_hdf5_parallel(data_filter, save_mesh, mesh_file_name,
                                   solution_file_name, MPI_COMM_WORLD);

      double time = parameters.time_step*snapshot_n;
      auto new_xdmf_entry = data_out.create_xdmf_entry
        (data_filter, mesh_file_name, solution_file_name, time, MPI_COMM_WORLD);
      xdmf_entries.push_back(std::move(new_xdmf_entry));
      data_out.write_xdmf_file(xdmf_entries, xdmf_filename, MPI_COMM_WORLD);
    }
}
