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

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/bundled/boost/math/special_functions/round.hpp>

#include <memory>
#include <utility>
#include <vector>

#include "filter.h"
#include "parameters.h"
#include "../h5/h5.h"
#include "../pod/pod.h"
#include "../ode/ode.h"
#include "rk_factory.h"
#include "ns.h"

namespace NavierStokes
{
  using namespace dealii;


  template<int dim>
  class ROM
  {
  public:
    ROM();
    void run();

  private:
    void setup_vectors_and_dof_handler();
    void setup_reduced_system();
    void time_iterate();

    std::unique_ptr<FE_Q<dim>>       fe;
    std::unique_ptr<QGauss<dim>>     quad;
    Triangulation<dim>               triangulation;
    SparsityPattern                  sparsity_pattern;
    std::shared_ptr<DoFHandler<dim>> dof_handler;

    FullMatrix<double>               mass_matrix;
    FullMatrix<double>               boundary_matrix;
    FullMatrix<double>               laplace_matrix;
    FullMatrix<double>               linear_operator;
    FullMatrix<double>               joint_convection;
    std::vector<FullMatrix<double>>  nonlinear_operator;
    Vector<double>                   mean_contribution_vector;

    unsigned int                     n_dofs;
    unsigned int                     n_pod_dofs;

    std::shared_ptr<std::vector<BlockVector<double>>> pod_vectors;
    std::shared_ptr<BlockVector<double>>              mean_vector;
    std::shared_ptr<std::vector<BlockVector<double>>> filtered_pod_vectors;
    std::shared_ptr<BlockVector<double>>              filtered_mean_vector;
    Vector<double>                   solution;

    double                           time;
    unsigned int                     timestep_number;

    POD::NavierStokes::Parameters    parameters;

    POD::PODOutput<dim>              pod_output;
  };


  template<int dim>
  ROM<dim>::ROM()
    :
    fe {new FE_Q<dim>(2)},
    quad {new QGauss<dim>(2)},
    dof_handler {new DoFHandler<dim>},
    pod_vectors {new std::vector<BlockVector<double>>},
    mean_vector {new BlockVector<double>},
    filtered_pod_vectors {new std::vector<BlockVector<double>>},
    filtered_mean_vector {new BlockVector<double>},
    timestep_number {0}
  {
    parameters.read_data(std::string("parameter-file.prm"));
    time = parameters.initial_time;
    // 2*N - 1 = 3*D -> N should be at least (3*D + 2)/2
    fe = std::unique_ptr<FE_Q<dim>>(new FE_Q<dim>(parameters.fe_order));
    quad = std::unique_ptr<QGauss<dim>>(new QGauss<dim>((3*fe->degree + 2)/2));
  }


  template<int dim>
  void ROM<dim>::setup_vectors_and_dof_handler()
  {
    // TODO un-hardcode these values
    POD::load_pod_basis("pod-vector-*.h5", "mean-vector.h5", *mean_vector,
                        *pod_vectors);
    POD::create_dof_handler_from_triangulation_file
    ("triangulation.txt", parameters.renumber, *fe, *dof_handler, triangulation);

    n_dofs = pod_vectors->at(0).block(0).size();
    n_pod_dofs = pod_vectors->size();

    {
      DynamicSparsityPattern d_sparsity(dof_handler->n_dofs());
      DoFTools::make_sparsity_pattern(*dof_handler, d_sparsity);
      sparsity_pattern.copy_from(d_sparsity);
    }
    // This is an abuse of notation to save duplication: if the POD vectors are
    // not filtered, then simply assign the filtered pod vectors pointer to
    // point to the unfiltered ones.
    if ((parameters.filter_model == POD::FilterModel::Differential
         or parameters.filter_model == POD::FilterModel::LerayHybrid)
        and parameters.filter_radius != 0.0)
      {
        std::shared_ptr<SparseMatrix<double>> full_mass_matrix
        {new SparseMatrix<double>};
        full_mass_matrix->reinit(sparsity_pattern);
        SparseMatrix<double> full_laplace_matrix(sparsity_pattern);
        SparseMatrix<double> full_boundary_matrix(sparsity_pattern);
        QGauss<dim - 1> face_quad(fe->degree + 3);
        MatrixCreator::create_mass_matrix(*dof_handler, *quad, *full_mass_matrix);
        MatrixCreator::create_laplace_matrix
        (*dof_handler, *quad, full_laplace_matrix);
        POD::NavierStokes::create_boundary_matrix
        (*dof_handler, face_quad, parameters.outflow_label, full_boundary_matrix);

        Leray::LerayFilter filter
        (parameters.filter_radius, full_mass_matrix, full_boundary_matrix,
         full_laplace_matrix);

        if (parameters.filter_mean
            or parameters.filter_model == POD::FilterModel::LerayHybrid)
          {
            filter.apply(*filtered_mean_vector, *mean_vector);
          }
        else
          {
            filtered_mean_vector = mean_vector;
          }

        if (parameters.filter_model == POD::FilterModel::Differential)
          {
            filtered_pod_vectors->resize(n_pod_dofs);
            for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs;
                 ++pod_vector_n)
              {
                filter.apply(filtered_pod_vectors->at(pod_vector_n),
                             pod_vectors->at(pod_vector_n));
              }
          }
        else
          {
            filtered_pod_vectors = pod_vectors;
          }
        std::cout << "finished filtering." << std::endl;
      }
    else
      {
        filtered_pod_vectors = pod_vectors;
        filtered_mean_vector = mean_vector;
      }
  }


  template<int dim>
  void ROM<dim>::setup_reduced_system()
  {
    FullMatrix<double> convection_matrix_0;
    FullMatrix<double> convection_matrix_1;

    {
      SparseMatrix<double> full_mass_matrix(sparsity_pattern);
      MatrixCreator::create_mass_matrix(*dof_handler, *quad, full_mass_matrix);
      POD::create_reduced_matrix(*pod_vectors, full_mass_matrix, mass_matrix);

      BlockVector<double> centered_initial;
      std::string initial("initial.h5");
      H5::load_block_vector(initial, centered_initial);
      solution.reinit(n_pod_dofs);
      centered_initial -= *mean_vector;
      for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
        {
          Vector<double> temp(n_dofs);
          full_mass_matrix.vmult(temp, centered_initial.block(dim_n));
          for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs;
               ++pod_vector_n)
            {
              solution[pod_vector_n] +=
                temp * pod_vectors->at(pod_vector_n).block(dim_n);
            }
        }
    }
    std::cout << "assembled the reduced mass matrix." << std::endl;

    {
      laplace_matrix.reinit(n_pod_dofs, n_pod_dofs);
      SparseMatrix<double> full_laplace_matrix(sparsity_pattern);
      MatrixCreator::create_laplace_matrix(*dof_handler, *quad, full_laplace_matrix);
      POD::create_reduced_matrix(*pod_vectors, full_laplace_matrix, laplace_matrix);

      mean_contribution_vector.reinit(n_pod_dofs);
      for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
        {
          Vector<double> temp(n_dofs);
          full_laplace_matrix.vmult(temp, mean_vector->block(dim_n));
          for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs; ++pod_vector_n)
            {
              mean_contribution_vector(pod_vector_n) -=
                1.0/parameters.reynolds_n*(temp * pod_vectors->at(pod_vector_n).block(dim_n));
            }
        }
    }
    std::cout << "assembled the reduced laplace matrix." << std::endl;

    {
      SparseMatrix<double> full_boundary_matrix(sparsity_pattern);
      QGauss<dim - 1> face_quad(fe->degree + 3);
      POD::NavierStokes::create_boundary_matrix
      (*dof_handler, face_quad, parameters.outflow_label, full_boundary_matrix);

      std::vector<unsigned int> dims {0};
      POD::create_reduced_matrix(*pod_vectors, full_boundary_matrix, dims, boundary_matrix);

      Vector<double> temp(n_dofs);
      full_boundary_matrix.vmult(temp, mean_vector->block(0));
      for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs; ++pod_vector_n)
        {
          mean_contribution_vector(pod_vector_n) +=
            1.0/parameters.reynolds_n*(temp * pod_vectors->at(pod_vector_n).block(0));
        }
    }
    std::cout << "assembled the reduced boundary matrix." << std::endl;

    // Note that, without filtering, filtered_pod_vectors and pod_vectors point
    // to the same thing. Same with the mean vectors.
    POD::NavierStokes::create_reduced_advective_linearization
    (*dof_handler, sparsity_pattern, *quad, *filtered_mean_vector, *pod_vectors,
     convection_matrix_0);
    POD::NavierStokes::create_reduced_gradient_linearization
    (*dof_handler, sparsity_pattern, *quad, *mean_vector, *pod_vectors,
     *filtered_pod_vectors, convection_matrix_1);
    std::cout << "assembled the two convection matrices." << std::endl;

    Vector<double> nonlinear_contribution(n_pod_dofs);
    POD::NavierStokes::create_nonlinear_centered_contribution
    (*dof_handler, sparsity_pattern, *quad, *filtered_mean_vector, *mean_vector,
     *pod_vectors, nonlinear_contribution);
    mean_contribution_vector.add(-1.0, nonlinear_contribution);

    // The joint convection matrix is necessary for the L2 Projection model (all
    // terms resulting from the nonlinearity must be filtered)
    joint_convection.reinit(n_pod_dofs, n_pod_dofs);
    joint_convection.add(-1.0, convection_matrix_0);
    joint_convection.add(-1.0, convection_matrix_1);

    linear_operator.reinit(n_pod_dofs, n_pod_dofs);
    linear_operator.add(-1.0/parameters.reynolds_n, laplace_matrix);
    linear_operator.add(1.0/parameters.reynolds_n, boundary_matrix);
    linear_operator.add(-1.0, convection_matrix_0);
    linear_operator.add(-1.0, convection_matrix_1);
    std::cout << "assembled all affine terms." << std::endl;

    POD::NavierStokes::create_reduced_nonlinearity
    (*dof_handler, sparsity_pattern, *quad, *pod_vectors, *filtered_pod_vectors,
     nonlinear_operator);
    std::cout << "assembled the nonlinearity." << std::endl;
  }


  template<int dim>
  void ROM<dim>::time_iterate()
  {
    Vector<double> old_solution(solution);

    std::string outname;
    std::unique_ptr<ODE::RungeKuttaBase> rk_method {new ODE::RungeKutta4()};
    std::tie(outname, rk_method) = POD::NavierStokes::rk_factory
      (boundary_matrix, joint_convection, laplace_matrix,
       linear_operator, mean_contribution_vector, mass_matrix,
       nonlinear_operator, parameters);

    int n_save_steps = boost::math::iround
      ((parameters.final_time - parameters.initial_time)/parameters.time_step)
      /parameters.output_interval;
    FullMatrix<double> solutions(n_save_steps + 1, n_pod_dofs);
    unsigned int output_n = 0;
    if (parameters.save_plot_pictures)
      {
        auto prefix = outname.substr(0, outname.size() - 3) + "-";
        pod_output.reinit(dof_handler, mean_vector, pod_vectors, prefix);
      }
    while (time < parameters.final_time)
      {
        old_solution = solution;
        rk_method->step(parameters.time_step, old_solution, solution);

        if (timestep_number % parameters.output_interval == 0)
          {
            for (unsigned int i = 0; i < n_pod_dofs; ++i)
              {
                solutions(output_n, i) = solution(i);
              }
            ++output_n;
            if (parameters.save_plot_pictures
                and time >= parameters.output_time_start
                and time <= parameters.output_time_stop)
              {
                pod_output.save_solution(solution, time, timestep_number);
              }
          }
        ++timestep_number;
        time += parameters.time_step;
      }

    H5::save_full_matrix(outname, solutions);
  }


  template<int dim>
  void ROM<dim>::run()
  {
    setup_vectors_and_dof_handler();
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

        ROM<3> nse_solver;
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
