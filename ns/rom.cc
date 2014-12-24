/* ---------------------------------------------------------------------
 * $Id: rom.cc $
 *
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

#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/bundled/boost/archive/text_iarchive.hpp>
// needed to get around the "save the dof handler issue"
#include <deal.II/dofs/dof_faces.h>
#include <deal.II/dofs/dof_levels.h>

#include <array>
#include <fstream>
#include <iostream>
#include <glob.h>
#include <limits>
#include <memory>

#include "../h5/h5.h"
#include "../pod/pod.h"
#include "../ode/ode.h"
#include "ns.h"

constexpr int outflow_label = 3;
constexpr int patch_refinement = 2;
constexpr int output_interval = 1000;

constexpr double re = 100.0;
constexpr double initial_time = 20.0;
constexpr double final_time = 40.0;
constexpr double time_step = 5.0e-5;

namespace NavierStokes
{
  using namespace dealii;
  using POD::NavierStokes::trilinearity_term;
  using POD::NavierStokes::create_boundary_matrix;


  template<int dim>
  class ROM
  {
  public:
    ROM(bool renumber);
    void run();

  private:
    void load_mesh();
    void load_vectors();
    void setup_linear();
    void setup_nonlinear();
    void time_iterate();
    void output_results();

    FE_Q<dim>                        fe;
    QGauss<dim>                      quad;
    Triangulation<dim>               triangulation;
    DoFHandler<dim>                  dof_handler;
    bool                             renumber;

    FullMatrix<double>               mass_matrix;
    FullMatrix<double>               linear_operator;
    std::vector<FullMatrix<double>>  nonlinear_operator;
    Vector<double>                   mean_contribution_vector;

    unsigned int                     n_dofs;
    unsigned int                     n_pod_dofs;

    std::vector<BlockVector<double>> pod_vectors;
    BlockVector<double>              mean_vector;
    Vector<double>                   solution;

    double                           time;
    unsigned int                     timestep_number;
    double                           reynolds_n;

    std::vector<XDMFEntry>           xdmf_entries;
    bool                             write_mesh;
  };


  template<int dim>
  ROM<dim>::ROM(bool renumber)
    :
    fe(2), // TODO don't hardcode this: rely on some input file.
    quad(fe.degree + 3),
    renumber(renumber),
    time(initial_time),
    timestep_number(0),
    // TODO unhardcode this.
    reynolds_n(re),
    write_mesh(true)
  {}


  template<int dim>
  void ROM<dim>::load_mesh()
  {
    std::string triangulation_file_name = "triangulation.txt";
    std::filebuf file_buffer;
    file_buffer.open (triangulation_file_name, std::ios::in);
    std::istream in_stream (&file_buffer);
    boost::archive::text_iarchive archive(in_stream);
    archive >> triangulation;
    dof_handler.initialize(triangulation, fe);
    if (renumber)
      {
        std::cout << "renumbering." << std::endl;
        DoFRenumbering::boost::Cuthill_McKee (dof_handler);
      }
  }


  template<int dim>
  void ROM<dim>::load_vectors()
  {
    // TODO un-hardcode these values
    std::string pod_vector_glob("pod-vector-*.h5");
    std::string mean_vector_file_name("mean-vector.h5");

    glob_t glob_result;
    glob(pod_vector_glob.c_str(), GLOB_TILDE, nullptr, &glob_result);
    pod_vectors.resize(glob_result.gl_pathc);
    for (unsigned int i = 0; i < glob_result.gl_pathc; ++i)
      {
        BlockVector<double> pod_vector;
        std::string file_name(glob_result.gl_pathv[i]);
        auto start_number = file_name.find_first_of('0');
        auto end_number = file_name.find_first_of('.');
        unsigned int pod_vector_n = Utilities::string_to_int
                                    (file_name.substr(start_number, end_number - start_number));
        H5::load_block_vector(file_name, pod_vector);
        pod_vectors[pod_vector_n] = std::move(pod_vector);
      }
    H5::load_block_vector(mean_vector_file_name, mean_vector);
    n_dofs = pod_vectors[0].block(0).size();
    n_pod_dofs = pod_vectors.size();
    globfree(&glob_result);
  }


  template<int dim>
  void ROM<dim>::setup_linear()
  {
    FullMatrix<double> laplace_matrix;
    FullMatrix<double> boundary_matrix;
    FullMatrix<double> convection_matrix_0;
    FullMatrix<double> convection_matrix_1;
    std::cout << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, c_sparsity);
    SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(c_sparsity);

    {
      SparseMatrix<double> full_mass_matrix(sparsity_pattern);
      MatrixCreator::create_mass_matrix(dof_handler, quad, full_mass_matrix);
      POD::create_reduced_matrix(pod_vectors, full_mass_matrix, mass_matrix);

      BlockVector<double> centered_initial;
      std::string initial("initial.h5");
      H5::load_block_vector(initial, centered_initial);
      solution.reinit(n_pod_dofs);
      centered_initial -= mean_vector;
      for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
        {
          Vector<double> temp(n_dofs);
          for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs; ++ pod_vector_n)
            {
              full_mass_matrix.vmult(temp, centered_initial.block(dim_n));
              solution[pod_vector_n] += temp * pod_vectors[pod_vector_n].block(dim_n);
            }
        }
    }
    std::cout << "assembled the reduced mass matrix." << std::endl;

    {
      laplace_matrix.reinit(n_pod_dofs, n_pod_dofs);
      SparseMatrix<double> full_laplace_matrix(sparsity_pattern);
      MatrixCreator::create_laplace_matrix(dof_handler, quad, full_laplace_matrix);
      POD::create_reduced_matrix(pod_vectors, full_laplace_matrix, laplace_matrix);

      mean_contribution_vector.reinit(n_pod_dofs);
      for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
        {
          Vector<double> temp(n_dofs);
          full_laplace_matrix.vmult(temp, mean_vector.block(dim_n));
          for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs; ++pod_vector_n)
            {
              mean_contribution_vector(pod_vector_n) -=
                1.0/reynolds_n*(temp * pod_vectors[pod_vector_n].block(dim_n));
            }
        }
    }
    std::cout << "assembled the reduced laplace matrix." << std::endl;

    {
      SparseMatrix<double> full_boundary_matrix(sparsity_pattern);
      QGauss<dim - 1> face_quad(fe.degree + 3);
      create_boundary_matrix(dof_handler, face_quad, outflow_label, full_boundary_matrix);

      std::vector<unsigned int> dims {0};
      POD::create_reduced_matrix(pod_vectors, full_boundary_matrix, dims, boundary_matrix);

      Vector<double> temp(n_dofs);
      full_boundary_matrix.vmult(temp, mean_vector.block(0));
      for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs; ++pod_vector_n)
        {
          mean_contribution_vector(pod_vector_n) +=
            1.0/reynolds_n*(temp * pod_vectors[pod_vector_n].block(0));
        }
    }
    std::cout << "assembled the reduced boundary matrix." << std::endl;

    {
      convection_matrix_0.reinit(n_pod_dofs, n_pod_dofs);
      convection_matrix_1.reinit(n_pod_dofs, n_pod_dofs);
      for (unsigned int i = 0; i < n_pod_dofs; ++i)
        {
          // an OMP parallel for loop could *probably* be put here.
          for (unsigned int j = 0; j < n_pod_dofs; ++j)
            {
              convection_matrix_0(i, j) =
                trilinearity_term(quad, dof_handler, pod_vectors.at(i),
                                  mean_vector, pod_vectors.at(j));
              convection_matrix_1(i, j) =
                trilinearity_term(quad, dof_handler, pod_vectors.at(i),
                                  pod_vectors.at(j), mean_vector);
            }
        }
    }
    std::cout << "assembled the two convection matrices." << std::endl;
    // add on the boundary contributions from the convection term here.
    for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs; ++pod_vector_n)
      {
        mean_contribution_vector(pod_vector_n) -=
          trilinearity_term(quad, dof_handler, pod_vectors.at(pod_vector_n),
                            mean_vector, mean_vector);
      }

    linear_operator.reinit(n_pod_dofs, n_pod_dofs);
    linear_operator.add(-1.0/reynolds_n, laplace_matrix);
    linear_operator.add(1.0/reynolds_n, boundary_matrix);
    linear_operator.add(-1.0, convection_matrix_0);
    linear_operator.add(-1.0, convection_matrix_1);
  }


  template<int dim>
  void ROM<dim>::setup_nonlinear()
  {
    for (unsigned int i = 0; i < n_pod_dofs; ++i)
      {
        nonlinear_operator.emplace_back(n_pod_dofs);
        std::cout << "nonlinearity.size() = "
                  << nonlinear_operator.size()
                  << std::endl;
        #pragma omp parallel for
        for (unsigned int j = 0; j < n_pod_dofs; ++j)
          {
            for (unsigned int k = 0; k < n_pod_dofs; ++k)
              {
                nonlinear_operator[i](j, k) =
                  trilinearity_term(quad, dof_handler, pod_vectors.at(i),
                                    pod_vectors.at(j), pod_vectors.at(k));
              }
          }
      }
    std::cout << "assembled the tensor." << std::endl;
  }


  export template<int dim>
  void ROM<dim>::time_iterate()
  {
    Vector<double> old_solution(solution);
    std::unique_ptr<POD::NavierStokes::NavierStokesRHS>
    rhs_function(new POD::NavierStokes::NavierStokesRHS(linear_operator,
                                                        mass_matrix,
                                                        nonlinear_operator,
                                                        mean_contribution_vector));
    ODE::RungeKutta4 rk_method(std::move(rhs_function));
    while (time < final_time)
      {
        old_solution = solution;
        rk_method.step(time_step, old_solution, solution);

        if (timestep_number % output_interval == 0)
          {
            output_results();
          }
        ++timestep_number;
        time += time_step;
      }
  }


  template<int dim>
  void ROM<dim>::output_results()
  {
    std::string pod_file_name = "pod-solution-"
                                + Utilities::int_to_string(timestep_number, 10)
                                + ".h5";
    {
      BlockVector<double> solution_block(1, n_pod_dofs);
      solution_block.block(0) = solution;
      H5::save_block_vector(pod_file_name, solution_block);
    }

    BlockVector<double> fe_solution(dim, n_dofs);
    fe_solution = mean_vector;
    for (unsigned int i = 0; i < n_pod_dofs; ++i)
      {
        fe_solution.add(solution(i), pod_vectors[i]);
      }

    // save the data in a plot-friendly way too.
    std::vector<std::string> solution_names(dim, "v");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation
    (dim, DataComponentInterpretation::component_is_part_of_vector);

    FESystem<dim> vector_fe(fe, dim);
    DoFHandler<dim> vector_dof_handler(triangulation);
    vector_dof_handler.distribute_dofs(vector_fe);

    dealii::Vector<double> vector_solution (vector_dof_handler.n_dofs());
    std::vector<types::global_dof_index> loc_vector_dof_indices (vector_fe.dofs_per_cell),
        loc_component_dof_indices (fe.dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
    vector_cell = vector_dof_handler.begin_active(),
    vector_endc = vector_dof_handler.end(),
    component_cell = dof_handler.begin_active();
    for (; vector_cell != vector_endc; ++vector_cell, ++component_cell)
      {
        vector_cell->get_dof_indices(loc_vector_dof_indices);
        component_cell->get_dof_indices(loc_component_dof_indices);
        for (unsigned int j = 0; j < vector_fe.dofs_per_cell; ++j)
          {
            switch (vector_fe.system_to_base_index(j).first.first)
              {
              // TODO this is sloppy cut-and-paste from step-35
              case 0:
                vector_solution(loc_vector_dof_indices[j]) =
                  fe_solution.block(vector_fe.system_to_base_index(j).first.second)
                  (loc_component_dof_indices[vector_fe.system_to_base_index(j).second]);
                break;
              default:
                ExcInternalError();
              }
          }
      }
    DataOut<dim> data_out;
    data_out.attach_dof_handler(vector_dof_handler);

    data_out.add_data_vector(vector_solution, solution_names,
                             DataOut<dim>::type_dof_data,
                             component_interpretation);
    data_out.build_patches(patch_refinement);
    std::string solution_file_name = "solution-"
                                     + Utilities::int_to_string(timestep_number, 10) + ".h5";
    std::string mesh_file_name = "mesh.h5";
    std::string xdmf_filename = "solution.xdmf";

    DataOutBase::DataOutFilter data_filter
    (DataOutBase::DataOutFilterFlags(true, true));
    data_out.write_filtered_data(data_filter);
    data_out.write_hdf5_parallel(data_filter, write_mesh, mesh_file_name,
                                 solution_file_name, MPI_COMM_WORLD);

    // only save the mesh once.
    write_mesh = false;
    auto new_xdmf_entry = data_out.create_xdmf_entry
                          (data_filter, mesh_file_name, solution_file_name, time, MPI_COMM_WORLD);
    xdmf_entries.push_back(std::move(new_xdmf_entry));
    data_out.write_xdmf_file(xdmf_entries, xdmf_filename, MPI_COMM_WORLD);
  }


  template<int dim>
  void ROM<dim>::run()
  {
    load_mesh();
    load_vectors();
    setup_linear();
    setup_nonlinear();
    time_iterate();
  }
}


int main(int argc, char **argv)
{
  try
    {
      using namespace dealii;
      using namespace NavierStokes;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      {
        deallog.depth_console(0);

        ROM<2> nse_solver(false);
        nse_solver.run();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
