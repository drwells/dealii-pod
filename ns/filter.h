#ifndef __deal2_leray_filter_h
#define __deal2_leray_filter_h

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/vector.h>

#include <memory>

using namespace dealii;
namespace Leray
{
  class LerayFilter
  {
  public:
    LerayFilter(const double filter_radius,
                std::shared_ptr<SparseMatrix<double>> mass_matrix,
                SparseMatrix<double> &laplace_matrix,
                SparseMatrix<double> &boundary_matrix);
    void apply(BlockVector<double> &dst, const BlockVector<double> &src);
  private:
    const double filter_radius;
    std::shared_ptr<SparseMatrix<double>> mass_matrix;
    SparseMatrix<double> x_system_matrix;
    SparseMatrix<double> other_system_matrix;
    SparseILU<double> x_preconditioner;
    PreconditionChebyshev<> other_preconditioner;
  };
}
#endif
