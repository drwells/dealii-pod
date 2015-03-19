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
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/matrix_tools.h>

#include <algorithm>
#include <iostream>
#include <math.h>
#include <vector>

#include "../extra/extra.h"
#include "../h5/h5.h"
#include "../pod/pod.h"

constexpr int dim {3};
constexpr bool renumber {false};

using namespace dealii;
int main(int argc, char **argv)
  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

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

    SparsityPattern sparsity_pattern;
    {
      CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, c_sparsity);
      sparsity_pattern.copy_from(c_sparsity);
    }
    SparseMatrix<double> full_mass_matrix(sparsity_pattern);
    MatrixCreator::create_mass_matrix(dof_handler, quad, full_mass_matrix);

    auto file_names = extra::expand_file_names("snapshot-*h5");
    FullMatrix<double> pod_coefficients_matrix(file_names.size(), n_pod_vectors);
    BlockVector<double> fluctuation_norms(1, file_names.size());
    auto &fluctuations = fluctuation_norms.block(0);
    #pragma omp parallel for
    for (unsigned int snapshot_n = 0; snapshot_n < file_names.size(); ++snapshot_n)
      {
        auto &file_name = file_names[snapshot_n];
        BlockVector<double> snapshot;
        #pragma omp critical
       {
          H5::load_block_vector(file_name, snapshot);
        }
        snapshot -= mean_vector;

        Vector<double> temp(pod_vectors.at(0).block(0).size());
        for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
          {
            full_mass_matrix.vmult(temp, snapshot.block(dim_n));
            fluctuations[snapshot_n] += snapshot.block(dim_n) * temp;
            for (unsigned int pod_vector_n = 0; pod_vector_n < pod_vectors.size();
                 ++pod_vector_n)
              {
                pod_coefficients_matrix(snapshot_n, pod_vector_n) +=
                  temp * pod_vectors.at(pod_vector_n).block(dim_n);
              }
          }
        fluctuations[snapshot_n] = sqrt(fluctuations[snapshot_n]);
      }
    std::string projected_file_name("projected-pod-coefficients.h5");
    H5::save_full_matrix(projected_file_name, pod_coefficients_matrix);
    std::string fluctuation_norms_name("fluctuation-norms.h5");
    H5::save_block_vector(fluctuation_norms_name, fluctuation_norms);
  }
