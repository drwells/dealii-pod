/* ---------------------------------------------------------------------
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
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <algorithm>
#include <iostream>
#include <math.h>
#include <memory>
#include <string>
#include <vector>

#include "../extra/extra.h"
#include "../h5/h5.h"
#include "../pod/pod.h"
#include "../ns/ns.h"
#include "../ns/filter.h"

constexpr int dim {3};
constexpr bool renumber {false};
constexpr double fe_filter_radius {0.1};
constexpr double pod_filter_radius {0.067};
constexpr unsigned int outflow_label {3};
constexpr unsigned int patch_level {2};

using namespace dealii;
double get_l2_norm(const SparseMatrix<double> &mass_matrix,
                   const BlockVector<double> &solution)
{
    double result = 0;
    Vector<double> temp(solution.block(0).size());
    for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
      {
        mass_matrix.vmult(temp, solution.block(dim_n));
        result += temp * solution.block(dim_n);
      }
    return std::sqrt(result);
}


Vector<double> project_to_pod(const SparseMatrix<double> &mass_matrix,
                              const std::vector<BlockVector<double>> &pod_vectors,
                              const BlockVector<double> &solution)
{
  const unsigned int n_dofs = solution.block(0).size();
  const unsigned int n_pod_dofs = pod_vectors.size();

  Vector<double> result(n_pod_dofs);

  for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
    {
      Vector<double> temp(n_dofs);
      mass_matrix.vmult(temp, solution.block(dim_n));
      for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs;
           ++pod_vector_n)
        {
          result[pod_vector_n] +=
            temp * pod_vectors.at(pod_vector_n).block(dim_n);
        }
    }

  return result;
}


BlockVector<double> project_to_fe(const std::vector<BlockVector<double>> &pod_vectors,
                                  const Vector<double> coefficients)
{
  BlockVector<double> solution
    (pod_vectors.at(0).n_blocks(), pod_vectors.at(0).block(0).size());
  for (unsigned int i = 0; i < pod_vectors.size(); ++i)
    {
      solution.add(coefficients[i], pod_vectors.at(i));
    }

  return solution;
}


int main(int argc, char **argv)
  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 4);

    const FE_Q<dim> fe(2);
    const QGauss<dim> quad((3*fe.degree + 2)/2);

    std::vector<BlockVector<double>> pod_vectors;
    BlockVector<double> mean_vector;
    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;

    POD::create_dof_handler_from_triangulation_file
      ("triangulation.txt", renumber, fe, dof_handler, triangulation);
    POD::load_pod_basis("pod-vector-0*h5", "mean-vector.h5", mean_vector, pod_vectors);
    const unsigned int n_pod_vectors = pod_vectors.size();

    // load and sort the snapshot names.
    auto file_names = extra::expand_file_names("snapshot-*h5");

    // setup the FE filter.
    SparsityPattern sparsity_pattern;
    {
      CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, c_sparsity);
      sparsity_pattern.copy_from(c_sparsity);
    }

    std::shared_ptr<SparseMatrix<double>> full_mass_matrix
      {new SparseMatrix<double>(sparsity_pattern)};
    SparseMatrix<double> full_laplace_matrix(sparsity_pattern);
    SparseMatrix<double> full_boundary_matrix(sparsity_pattern);
    QGauss<dim - 1> face_quad(fe.degree + 3);
    MatrixCreator::create_mass_matrix(dof_handler, quad, *full_mass_matrix);
    MatrixCreator::create_laplace_matrix
      (dof_handler, quad, full_laplace_matrix);
    POD::NavierStokes::create_boundary_matrix
      (dof_handler, face_quad, outflow_label, full_boundary_matrix);

    Leray::LerayFilter filter
      (fe_filter_radius, full_mass_matrix, full_boundary_matrix,
       full_laplace_matrix);

    // filter the POD vectors.
    std::vector<BlockVector<double>> filtered_pod_vectors(n_pod_vectors);
    for (unsigned int i = 0; i < n_pod_vectors; ++i)
      {
        filtered_pod_vectors.at(i).reinit(pod_vectors.at(i).size());
        filter.apply(filtered_pod_vectors.at(i), pod_vectors.at(i));
      }

    // set up the POD filter.
    FullMatrix<double> mass_matrix;
    POD::create_reduced_matrix(pod_vectors, *full_mass_matrix, mass_matrix);
    FullMatrix<double> laplace_matrix;
    POD::create_reduced_matrix(pod_vectors, full_laplace_matrix, laplace_matrix);
    FullMatrix<double> boundary_matrix;
    POD::create_reduced_matrix(pod_vectors, full_boundary_matrix, boundary_matrix);
    LAPACKFullMatrix<double> pod_filter;
    pod_filter.reinit(n_pod_vectors);
    {
      FullMatrix<double> temp(n_pod_vectors);
      temp = mass_matrix;
      temp.add(pod_filter_radius*pod_filter_radius, laplace_matrix);
      temp.add(-1.0*pod_filter_radius*pod_filter_radius, boundary_matrix);
      pod_filter = temp;
    }
    pod_filter.compute_lu_factorization();

    // for simplicity, this only saves the y velocity.
    std::vector<XDMFEntry> xdmf_entries;
    std::string xdmf_filename = "projections.xdmf";
    std::string mesh_file_name = "mesh.h5";
    for (unsigned int snapshot_n = 0; snapshot_n < file_names.size(); ++snapshot_n)
      {
        auto &file_name = file_names.at(snapshot_n);
        BlockVector<double> snapshot;
        H5::load_block_vector(file_name, snapshot);
        snapshot -= mean_vector;
        BlockVector<double> filtered_snapshot;

        for (unsigned int offset_n = 0; offset_n < 4; ++offset_n)
          {
            switch (offset_n)
              {
              case 0: // save snapshot as-is
                {
                  filtered_snapshot = snapshot;
                  std::cout << "fluctuation norm: "
                            << get_l2_norm(*full_mass_matrix, filtered_snapshot)
                            << std::endl;
                }
                break;
              case 1: // the POD option for filtering
                {
                  auto filtered_coefficients = project_to_pod
                    (*full_mass_matrix, pod_vectors, snapshot);
                  {
                    Vector<double> temp(filtered_coefficients.size());
                    mass_matrix.vmult(temp, filtered_coefficients);
                    filtered_coefficients = temp;
                  }
                  std::cout << filtered_coefficients.l2_norm() << std::endl;
                  pod_filter.apply_lu_factorization(filtered_coefficients, false);
                  std::cout << filtered_coefficients.l2_norm() << std::endl;
                  filtered_snapshot = project_to_fe
                    (pod_vectors, filtered_coefficients);
                  std::cout << "POD-option norm: "
                            << get_l2_norm(*full_mass_matrix, filtered_snapshot)
                            << std::endl;
                }
                break;
              case 2: // the FE option for filtering
                {
                  auto coefficients = project_to_pod
                    (*full_mass_matrix, pod_vectors, snapshot);
                  filtered_snapshot = project_to_fe
                    (filtered_pod_vectors, coefficients);
                  std::cout << "FE-option norm: "
                            << get_l2_norm(*full_mass_matrix, filtered_snapshot)
                            << std::endl;
                }
                break;
              case 3: // no filtering
                {
                  auto coefficients = project_to_pod
                    (*full_mass_matrix, pod_vectors, snapshot);
                  filtered_snapshot = project_to_fe(pod_vectors, coefficients);
                  std::cout << "projected norm: "
                            << get_l2_norm(*full_mass_matrix, filtered_snapshot)
                            << std::endl;
                  std::cout << "projected norm: "
                            << coefficients.l2_norm()
                            << std::endl;
                }
                break;
              default:
                StandardExceptions::ExcNotImplemented();
              }
            std::string solution_file_name = "y-velocity-"
              + extra::int_to_string(10*snapshot_n + offset_n, 10)
              + ".h5";

            DataOut<dim> data_out;
            data_out.attach_dof_handler(dof_handler);
            std::vector<std::string> solution_name(1, "v");
            std::vector<DataComponentInterpretation::DataComponentInterpretation>
              component_interpretation(1, DataComponentInterpretation::component_is_scalar);
            data_out.add_data_vector(filtered_snapshot.block(1), solution_name,
                                     DataOut<dim>::type_dof_data, component_interpretation);
            data_out.build_patches(patch_level);
            bool save_mesh = false;
            if (snapshot_n == 0 and offset_n == 0)
              {
                save_mesh = true;
              }

            DataOutBase::DataOutFilter data_filter
              (DataOutBase::DataOutFilterFlags(true, true));
            data_out.write_filtered_data(data_filter);
            data_out.write_hdf5_parallel(data_filter, save_mesh, mesh_file_name,
                                         solution_file_name, MPI_COMM_WORLD);

            auto time = static_cast<double>(10*snapshot_n + offset_n);
            auto new_xdmf_entry = data_out.create_xdmf_entry
              (data_filter, mesh_file_name, solution_file_name, time, MPI_COMM_WORLD);
            xdmf_entries.push_back(std::move(new_xdmf_entry));
            data_out.write_xdmf_file(xdmf_entries, xdmf_filename, MPI_COMM_WORLD);
          }
      }
  }
