#ifndef __deal2__pod_h
#define __deal2__pod_h
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <vector>

#include "../h5/h5.h"

using namespace dealii;
namespace POD
{
  class BlockPODBasis
  {
  public:
    BlockPODBasis();
    BlockPODBasis(unsigned int n_blocks, unsigned int n_dofs_per_block);

    void reinit(unsigned int n_blocks, unsigned int n_dofs_per_block);
    void project_load_vector(dealii::BlockVector<double> &load_vector,
                             dealii::BlockVector<double> &pod_load_vector) const;
    void project_to_fe(const dealii::BlockVector<double> &pod_vector,
                       dealii::BlockVector<double> &fe_vector) const;

    std::vector<dealii::BlockVector<double>> vectors;
    dealii::BlockVector<double> mean_vector;
    std::vector<double> singular_values;
    unsigned int get_n_pod_vectors() const;
  private:
    unsigned int n_blocks;
    unsigned int n_dofs_per_block;
  };

  void method_of_snapshots(dealii::SparseMatrix<double> &mass_matrix,
                           std::vector<std::string> &snapshot_file_names,
                           BlockPODBasis &pod_basis);

  void create_reduced_matrix(const std::vector<BlockVector<double>> &pod_vectors,
                             const SparseMatrix<double> &full_matrix,
                             FullMatrix<double> &rom_matrix);


  void create_reduced_matrix(const std::vector<BlockVector<double>> &pod_vectors,
                             const SparseMatrix<double> &full_matrix,
                             const std::vector<unsigned int> dims,
                             FullMatrix<double> &rom_matrix);

}
#endif
