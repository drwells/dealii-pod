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

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/bundled/boost/archive/text_iarchive.hpp>
// needed to get around the "save the dof handler issue"
#include <deal.II/dofs/dof_faces.h>
#include <deal.II/dofs/dof_levels.h>

#include <fstream>
#include <iostream>
#include <glob.h>
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
constexpr double filter_radius = 0.000;


constexpr bool save_plot_pictures = false;

namespace NavierStokes
{
  using namespace dealii;


  template<int dim>
  class ROM
  {
  public:
    ROM(bool renumber);
    void run();

  private:
    void load_mesh();
    void load_vectors();
    void setup_reduced_system();
    void time_iterate();

    FE_Q<dim>                        fe;
    QGauss<dim>                      quad;
    Triangulation<dim>               triangulation;
    SparsityPattern                  sparsity_pattern;
    std::shared_ptr<DoFHandler<dim>> dof_handler;
    bool                             renumber;

    FullMatrix<double>               mass_matrix;
    FullMatrix<double>               laplace_matrix;
    FullMatrix<double>               linear_operator;
    std::vector<FullMatrix<double>>  nonlinear_operator;
    Vector<double>                   mean_contribution_vector;

    unsigned int                     n_dofs;
    unsigned int                     n_pod_dofs;

    std::shared_ptr<std::vector<BlockVector<double>>> pod_vectors;
    std::shared_ptr<BlockVector<double>>              mean_vector;
    Vector<double>                   solution;

    double                           time;
    unsigned int                     timestep_number;
    double                           reynolds_n;

    bool                             output_initialized;
    POD::PODOutput<dim>              output;
  };


  template<int dim>
  ROM<dim>::ROM(bool renumber)
    :
    fe(2), // TODO don't hardcode this: rely on some input file.
    // 2*N - 1 = 3*D -> N should be at least (3*D + 2)/2
    quad((3*fe.degree + 2)/2),
    dof_handler {new DoFHandler<dim>},
  renumber(renumber),
  pod_vectors {new std::vector<BlockVector<double>>},
  mean_vector {new BlockVector<double>},
  time(initial_time),
  timestep_number(0),
  reynolds_n(re), // TODO unhardcode this.
  output_initialized(false)
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
    dof_handler->initialize(triangulation, fe);
    if (renumber)
      {
        std::cout << "renumbering." << std::endl;
        DoFRenumbering::boost::Cuthill_McKee (*dof_handler);
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
    pod_vectors->resize(glob_result.gl_pathc);
    for (unsigned int i = 0; i < glob_result.gl_pathc; ++i)
      {
        BlockVector<double> pod_vector;
        std::string file_name(glob_result.gl_pathv[i]);
        auto start_number = file_name.find_first_of('0');
        auto end_number = file_name.find_first_of('.');
        unsigned int pod_vector_n = Utilities::string_to_int
                                    (file_name.substr(start_number, end_number - start_number));
        H5::load_block_vector(file_name, pod_vector);
        pod_vectors->at(pod_vector_n) = std::move(pod_vector);
      }
    H5::load_block_vector(mean_vector_file_name, *mean_vector);
    n_dofs = pod_vectors->at(0).block(0).size();
    n_pod_dofs = pod_vectors->size();
    globfree(&glob_result);
  }


  template<int dim>
  void ROM<dim>::setup_reduced_system()
  {
    FullMatrix<double> boundary_matrix;
    FullMatrix<double> convection_matrix_0;
    FullMatrix<double> convection_matrix_1;
    std::cout << "DoFs: " << dof_handler->n_dofs() << std::endl;

    CompressedSparsityPattern c_sparsity(dof_handler->n_dofs());
    DoFTools::make_sparsity_pattern(*dof_handler, c_sparsity);
    sparsity_pattern.copy_from(c_sparsity);

    {
      SparseMatrix<double> full_mass_matrix(sparsity_pattern);
      MatrixCreator::create_mass_matrix(*dof_handler, quad, full_mass_matrix);
      POD::create_reduced_matrix(*pod_vectors, full_mass_matrix, mass_matrix);

      BlockVector<double> centered_initial;
      std::string initial("initial.h5");
      H5::load_block_vector(initial, centered_initial);
      solution.reinit(n_pod_dofs);
      centered_initial -= *mean_vector;
      for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
        {
          Vector<double> temp(n_dofs);
          for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs; ++ pod_vector_n)
            {
              full_mass_matrix.vmult(temp, centered_initial.block(dim_n));
              solution[pod_vector_n] += temp * pod_vectors->at(pod_vector_n).block(dim_n);
            }
        }
    }
    std::cout << "assembled the reduced mass matrix." << std::endl;

    {
      laplace_matrix.reinit(n_pod_dofs, n_pod_dofs);
      SparseMatrix<double> full_laplace_matrix(sparsity_pattern);
      MatrixCreator::create_laplace_matrix(*dof_handler, quad, full_laplace_matrix);
      POD::create_reduced_matrix(*pod_vectors, full_laplace_matrix, laplace_matrix);

      mean_contribution_vector.reinit(n_pod_dofs);
      for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
        {
          Vector<double> temp(n_dofs);
          full_laplace_matrix.vmult(temp, mean_vector->block(dim_n));
          for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs; ++pod_vector_n)
            {
              mean_contribution_vector(pod_vector_n) -=
                1.0/reynolds_n*(temp * pod_vectors->at(pod_vector_n).block(dim_n));
            }
        }
    }
    std::cout << "assembled the reduced laplace matrix." << std::endl;

    {
      SparseMatrix<double> full_boundary_matrix(sparsity_pattern);
      QGauss<dim - 1> face_quad(fe.degree + 3);
      POD::NavierStokes::create_boundary_matrix
      (*dof_handler, face_quad, outflow_label, full_boundary_matrix);

      std::vector<unsigned int> dims {0};
      POD::create_reduced_matrix(*pod_vectors, full_boundary_matrix, dims, boundary_matrix);

      Vector<double> temp(n_dofs);
      full_boundary_matrix.vmult(temp, mean_vector->block(0));
      for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs; ++pod_vector_n)
        {
          mean_contribution_vector(pod_vector_n) +=
            1.0/reynolds_n*(temp * pod_vectors->at(pod_vector_n).block(0));
        }
    }
    std::cout << "assembled the reduced boundary matrix." << std::endl;

    POD::NavierStokes::create_reduced_advective_linearization
      (*dof_handler, sparsity_pattern, quad, *mean_vector, *pod_vectors,
       convection_matrix_0);
    POD::NavierStokes::create_reduced_gradient_linearization
      (*dof_handler, sparsity_pattern, quad, *mean_vector, *pod_vectors,
       convection_matrix_1);
    std::cout << "assembled the two convection matrices." << std::endl;
    for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs; ++pod_vector_n)
      {
        mean_contribution_vector(pod_vector_n) -=
          POD::NavierStokes::trilinearity_term
          (quad, *dof_handler, pod_vectors->at(pod_vector_n), *mean_vector,
           *mean_vector);
      }

    linear_operator.reinit(n_pod_dofs, n_pod_dofs);
    linear_operator.add(-1.0/reynolds_n, laplace_matrix);
    linear_operator.add(1.0/reynolds_n, boundary_matrix);
    linear_operator.add(-1.0, convection_matrix_0);
    linear_operator.add(-1.0, convection_matrix_1);

    POD::NavierStokes::create_reduced_nonlinearity
    (*dof_handler, sparsity_pattern, quad, *pod_vectors, nonlinear_operator);
  }


  template<int dim>
  void ROM<dim>::time_iterate()
  {
    Vector<double> old_solution(solution);
    std::unique_ptr<POD::NavierStokes::NavierStokesLerayRegularizationRHS>
    rhs_function(new POD::NavierStokes::NavierStokesLerayRegularizationRHS
                 (linear_operator, mass_matrix, laplace_matrix, nonlinear_operator,
                  mean_contribution_vector, filter_radius));
    ODE::RungeKutta4 rk_method(std::move(rhs_function));
    while (time < final_time)
      {
        old_solution = solution;
        rk_method.step(time_step, old_solution, solution);

        if (timestep_number % output_interval == 0)
          {
            if (!output_initialized)
              {
                output_initialized = true;
                output.reinit(dof_handler, mean_vector, pod_vectors, "solution-");
              }
            output.save_solution(solution, time, timestep_number, save_plot_pictures);

          }
        ++timestep_number;
        time += time_step;
      }
  }


  template<int dim>
  void ROM<dim>::run()
  {
    load_mesh();
    load_vectors();
    setup_reduced_system();
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
