#ifndef __deal2__pod_h
#define __deal2__pod_h

#include <deal.II/lac/sparse_matrix.h>
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
    unsigned int get_num_pod_vectors() const;
    void project_load_vector(dealii::Vector<double> &, dealii::Vector<double> &);
    void project_to_fe(const dealii::Vector<double> &pod_vector,
                       dealii::Vector<double> &fe_vector) const;
  };

  void copy_vector(VectorBase &src, unsigned int start_src, unsigned int start_dst,
                   unsigned int total, VectorBase &dst);

  void pod_basis(dealii::SparseMatrix<double> &mass_matrix,
                 std::map<int, dealii::Vector<double>> &snapshots,
                 const unsigned int num_pod_vectors,
                 PODBasis &result);

  void reduced_matrix(dealii::SparseMatrix<double> &mass_matrix,
                      PODBasis &pod_basis, dealii::FullMatrix<double> &pod_mass_matrix);

}

#endif
