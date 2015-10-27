#include <deal.II-pod/ns/filter.h>

namespace POD
{
  using namespace dealii;

  namespace Leray
  {
    LerayFilter::LerayFilter
    (const double filter_radius,
     std::shared_ptr<SparseMatrix<double>> mass_matrix,
     const SparseMatrix<double> &boundary_matrix,
     const SparseMatrix<double> &laplace_matrix) :
      filter_radius {filter_radius},
      mass_matrix {mass_matrix}
    {
      x_system_matrix.reinit(mass_matrix->get_sparsity_pattern());
      x_system_matrix.copy_from(*mass_matrix);
      other_system_matrix.reinit(mass_matrix->get_sparsity_pattern());
      other_system_matrix.copy_from(*mass_matrix);
      x_system_matrix.add(filter_radius*filter_radius, laplace_matrix);
      x_system_matrix.add(-1.0*filter_radius*filter_radius, boundary_matrix);
      other_system_matrix.add(filter_radius*filter_radius, laplace_matrix);

      const double diagonal_strengthening = 100;
      x_preconditioner.initialize
        (x_system_matrix, SparseILU<double>::AdditionalData(diagonal_strengthening));
      other_preconditioner.initialize
        (other_system_matrix, PreconditionChebyshev<>::AdditionalData(2));
    }

    void LerayFilter::apply
    (BlockVector<double> &dst, const BlockVector<double> &src)
    {
      SolverControl solver_control(4000);
      SolverGMRES<Vector<double>> solver (solver_control);
      dst.reinit(src.n_blocks(), src.block(0).size());

      Vector<double> rhs(src.block(0).size());
      mass_matrix->vmult(rhs, src.block(0));
      solver.solve(x_system_matrix, dst.block(0), rhs, x_preconditioner);
      for (unsigned int dim_n = 1; dim_n < src.n_blocks(); ++dim_n)
        {
          mass_matrix->vmult(rhs, src.block(dim_n));
          solver.solve(other_system_matrix, dst.block(dim_n), rhs, other_preconditioner);
        }
    }
  }
}
