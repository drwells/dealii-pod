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
#ifndef dealii__rom_leray_filter_h
#define dealii__rom_leray_filter_h

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/vector.h>

#include <memory>

namespace Leray
{
  using namespace dealii;

  class LerayFilter
  {
  public:
    LerayFilter(const double                          filter_radius,
                std::shared_ptr<SparseMatrix<double>> mass_matrix,
                const SparseMatrix<double>            &laplace_matrix,
                const SparseMatrix<double>            &boundary_matrix);

    void apply(BlockVector<double>       &dst,
               const BlockVector<double> &src);

  private:
    const double                          filter_radius;
    std::shared_ptr<SparseMatrix<double>> mass_matrix;
    SparseMatrix<double>                  x_system_matrix;
    SparseMatrix<double>                  other_system_matrix;
    SparseILU<double>                     x_preconditioner;
    PreconditionChebyshev<>               other_preconditioner;
  };
}
#endif
