#ifndef __deal2__pod_h
#define __deal2__pod_h
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/petsc_full_matrix.h>
#include <deal.II/lac/petsc_matrix_free.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector_base.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include <complex>
#include <map>
#include <vector>

#include "../h5/h5.h"

namespace POD
{
  using namespace dealii::PETScWrappers;
  class EigenvalueMethod : public MatrixFree
  {
  public:
    EigenvalueMethod(dealii::SparseMatrix<double> &mass_matrix,
                      dealii::FullMatrix<double> &snapshots);
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

  class BlockPODBasis
  {
  public:
    BlockPODBasis();
    BlockPODBasis(unsigned int n_blocks, unsigned int n_dofs_per_block);
    std::vector<dealii::BlockVector<double>> vectors;
    dealii::BlockVector<double> mean_vector;
    std::vector<double> singular_values;
    unsigned int get_n_pod_vectors() const;
    void reinit(unsigned int n_blocks, unsigned int n_dofs_per_block);
    void project_load_vector(dealii::BlockVector<double> &load_vector,
                             dealii::BlockVector<double> &pod_load_vector) const;
    void project_to_fe(const dealii::BlockVector<double> &pod_vector,
                       dealii::BlockVector<double> &fe_vector) const;
  private:
    unsigned int n_blocks;
    unsigned int n_dofs_per_block;
  };

  void copy_vector(VectorBase &src, unsigned int start_src, unsigned int start_dst,
                   unsigned int total, VectorBase &dst);

  void pod_basis(dealii::SparseMatrix<double> &mass_matrix,
                 dealii::FullMatrix<double> &snapshot_matrix,
                 const unsigned int num_pod_vectors,
                 PODBasis &result);

  void reduced_matrix(dealii::SparseMatrix<double> &mass_matrix,
                      PODBasis &pod_basis, dealii::FullMatrix<double> &pod_mass_matrix);

  void method_of_snapshots(dealii::SparseMatrix<double> &mass_matrix,
                           std::vector<std::string> &snapshot_file_names,
                           unsigned int n_pod_vectors,
                           BlockPODBasis &pod_basis);

}
#endif
