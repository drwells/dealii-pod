#include "ns.h"

namespace POD
{
  namespace NavierStokes
  {
    void setup_reduced_matrix (const std::vector<BlockVector<double>> &pod_vectors,
                               const SparseMatrix<double> &full_matrix,
                               FullMatrix<double> &rom_matrix)
    {
      std::vector<unsigned int> dims;
      for (unsigned int i = 0; i < pod_vectors.at(0).n_blocks(); ++i)
        {
          dims.push_back(i);
        }
      setup_reduced_matrix(pod_vectors, full_matrix, dims, rom_matrix);
    }


    void setup_reduced_matrix (const std::vector<BlockVector<double>> &pod_vectors,
                               const SparseMatrix<double> &full_matrix,
                               const std::vector<unsigned int> dims,
                               FullMatrix<double> &rom_matrix)
    {
      const unsigned int n_dofs = pod_vectors[0].block(0).size();
      const unsigned int n_pod_dofs = pod_vectors.size();
      rom_matrix.reinit(n_pod_dofs, n_pod_dofs);
      rom_matrix = 0.0;
      Vector<double> temp(n_dofs);
      for (auto dim_n : dims)
        {
          for (unsigned int row = 0; row < n_pod_dofs; ++row)
            {
              auto &left_vector = pod_vectors[row].block(dim_n);
              for (unsigned int column = 0; column < n_pod_dofs; ++column)
                {
                  auto &right_vector = pod_vectors[column].block(dim_n);
                  full_matrix.vmult(temp, right_vector);
                  rom_matrix(row, column) += left_vector*temp;
                }
            }
        }
    }
  }
}
