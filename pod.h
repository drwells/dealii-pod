#ifndef __deal2__pod_h
#define __deal2__pod_h

#include <deal.II/lac/vector.h>
#include <deal.II/lac/petsc_matrix_free.h>
#include <deal.II/lac/petsc_vector_base.h>
#include <deal.II/lac/petsc_full_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/slepc_solver.h>

namespace POD
{
  using namespace dealii::PETScWrappers;
  class MethodOfSnapshots : public MatrixFree
  {
  public:
    MethodOfSnapshots(SparseMatrix &mass_matrix, FullMatrix &snapshots);
    void vmult(VectorBase &dst, const VectorBase &src) const;
    void vmult_add(VectorBase &dst, const VectorBase &src) const;
    void Tvmult(VectorBase &dst, const VectorBase &src) const;
    void Tvmult_add(VectorBase &dst, const VectorBase &src) const;
  };

  class PODBasis
  {
  public:
    std::map<int, dealii::Vector<double>> vectors;
    std::vector<double> singular_values;
  };

  void pod_basis(dealii::SparseMatrix<double>& mass_matrix,
                 std::map<int, dealii::Vector<double>>& snapshots,
                 const unsigned int num_pod_vectors,
                 PODBasis &result);
}

#endif