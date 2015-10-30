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
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/bundled/boost/archive/text_iarchive.hpp>
// needed to get around the "save the dof handler issue"
#include <deal.II/dofs/dof_faces.h>
#include <deal.II/dofs/dof_levels.h>

#include <algorithm>
#include <iostream>
#include <math.h>
#include <vector>
#include <memory>

#include "../h5/h5.h"
#include "../pod/pod.h"
#include "../ns/filter.h"
#include "../ns/ns.h"
#include "../extra/extra.h"
#include "parameters.h"

constexpr int dim {3};

using namespace dealii;
int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Parameters parameters;
  parameters.read_data("parameters.prm");

  const FE_Q<dim> fe(parameters.fe_order);
  Triangulation<dim> triangulation;
  DoFHandler<dim> dof_handler;
  SparsityPattern sparsity_pattern;
  QGauss<dim> quad((3*fe.degree + 2)/2);

  POD::create_dof_handler_from_triangulation_file
    (parameters.triangulation_file_name, parameters.renumber, fe, dof_handler,
     triangulation);

  std::vector<BlockVector<double>> pod_vectors;
  BlockVector<double> mean_vector;
  POD::load_pod_basis
    (parameters.pod_vector_glob, parameters.mean_vector_file_name, mean_vector,
     pod_vectors);
  {
    DynamicSparsityPattern d_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, d_sparsity);
    sparsity_pattern.copy_from(d_sparsity);
  }

  FullMatrix<double> mass_matrix;
  FullMatrix<double> laplace_matrix;
  FullMatrix<double> boundary_matrix;
  auto full_mass_matrix = std::make_shared<SparseMatrix<double>>
    (sparsity_pattern);
  auto full_laplace_matrix = std::make_shared<SparseMatrix<double>>
    (sparsity_pattern);
  auto full_boundary_matrix = std::make_shared<SparseMatrix<double>>
    (sparsity_pattern);
  MatrixCreator::create_mass_matrix(dof_handler, quad, *full_mass_matrix);
  POD::create_reduced_matrix(pod_vectors, *full_mass_matrix, mass_matrix);
  MatrixCreator::create_laplace_matrix(dof_handler, quad, *full_laplace_matrix);
  POD::create_reduced_matrix(pod_vectors, *full_laplace_matrix, laplace_matrix);
  QGauss<dim - 1> face_quad(fe.degree + 3);
  POD::NavierStokes::create_boundary_matrix
    (dof_handler, face_quad, parameters.outflow_label, *full_boundary_matrix);
  POD::create_reduced_matrix(pod_vectors, *full_boundary_matrix, boundary_matrix);

  LAPACKFullMatrix<double> filter_matrix(pod_vectors.size());
  {
    FullMatrix<double> temp(pod_vectors.size());
    temp.add(1.0, mass_matrix);
    temp.add(parameters.filter_radius*parameters.filter_radius, laplace_matrix);
    temp.add(-1.0*parameters.filter_radius*parameters.filter_radius, boundary_matrix);
    filter_matrix = temp;
  }
  Leray::LerayFilter filter
    (parameters.filter_radius, full_mass_matrix, *full_boundary_matrix,
     *full_laplace_matrix);
  auto fe_filtered_mean_vector = mean_vector;
  filter.apply(fe_filtered_mean_vector, mean_vector);

  // assemble the 'load vector'.
  Vector<double> rhs_vector(pod_vectors.size());
  for (unsigned int i = 0; i < pod_vectors.size(); ++i)
    {
      BlockVector<double> lhs_vector(3, pod_vectors.at(0).block(0).size());
      for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
        {
          full_mass_matrix->vmult
            (lhs_vector.block(dim_n), mean_vector.block(dim_n));
        }
      rhs_vector(i) = lhs_vector * pod_vectors.at(i);
    }

  auto solution = rhs_vector;
  filter_matrix.compute_lu_factorization();
  filter_matrix.apply_lu_factorization(solution, false);

  BlockVector<double> pod_projected_mean_vector
    (dim, pod_vectors.at(0).block(0).size());
  for (unsigned int i = 0; i < pod_vectors.size(); ++i)
    {
      pod_projected_mean_vector.add(solution[i], pod_vectors.at(i));
      std::cout << solution[i] << std::endl;
    }
  H5::save_block_vector("pod-filtered-mean-vector.h5", pod_projected_mean_vector);
  H5::save_block_vector("fe-filtered-mean-vector.h5", fe_filtered_mean_vector);
  // compute relevant L2 norms
  double mean_l2_norm {0.0};
  double projected_mean_l2_norm {0.0};
  double mean_difference {0.0};
  Vector<double> temp(pod_vectors.at(0).block(0).size());
  Vector<double> temp2(pod_vectors.at(0).block(0).size());
  for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
    {
      full_mass_matrix->vmult(temp, fe_filtered_mean_vector.block(dim_n));
      mean_l2_norm += fe_filtered_mean_vector.block(dim_n) * temp;
      full_mass_matrix->vmult(temp, pod_projected_mean_vector.block(dim_n));
      projected_mean_l2_norm += pod_projected_mean_vector.block(dim_n) * temp;
      temp2 = fe_filtered_mean_vector.block(dim_n);
      temp2 -= pod_projected_mean_vector.block(dim_n);
      full_mass_matrix->vmult(temp, temp2);
      mean_difference += temp * temp2;
    }
  std::cout << "projected mean L2 norm: "
            << std::sqrt(projected_mean_l2_norm)
            << std::endl;
  std::cout << "mean L2 norm: "
            << std::sqrt(mean_l2_norm)
            << std::endl;
  std::cout << "mean difference L2 norm: "
            << std::sqrt(mean_difference)
            << std::endl;
}
