#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/matrix_tools.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <deal.II-pod/pod/pod.h>
#include <deal.II-pod/h5/h5.h>
#include <deal.II-pod/extra/extra.h>

constexpr int dim {3};

int main(int argc, char **argv)
{
  using namespace dealii;
  using namespace POD;

  SparsityPattern sparsity_pattern;
  Triangulation<dim> triangulation;
  FE_Q<dim> fe(2);
  QGauss<dim> quad(4);
  DoFHandler<dim> dof_handler;
  SparseMatrix<double> mass_matrix;
  {
    POD::create_dof_handler_from_triangulation_file
      ("triangulation.txt", false, fe, dof_handler, triangulation);
    DynamicSparsityPattern d_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, d_sparsity);
    sparsity_pattern.copy_from(d_sparsity);
  }
  mass_matrix.reinit(sparsity_pattern);
  MatrixCreator::create_mass_matrix(dof_handler, quad, mass_matrix);
  std::vector<BlockVector<double>> pod_vectors;
  BlockVector<double> mean_vector;
  POD::load_pod_basis("pod-vector*h5", "mean-vector.h5", mean_vector,
                      pod_vectors);
  std::vector<double> projection_errors(pod_vectors.size(), 0.0);

  Vector<double> temp(mean_vector.block(0).size());
  for (auto &snapshot_name : extra::expand_file_names("snapshot*h5"))
    {
      BlockVector<double> snapshot;
      H5::load_block_vector(snapshot_name, snapshot);
      snapshot -= mean_vector;
      BlockVector<double> snapshot_residue(snapshot);
      for (unsigned int pod_vector_n = 0; pod_vector_n < pod_vectors.size();
           ++pod_vector_n)
        {
          double coefficient {0.0};
          for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
            {
              mass_matrix.vmult(temp, snapshot.block(dim_n));
              coefficient += temp * pod_vectors.at(pod_vector_n).block(dim_n);
            }
          snapshot_residue.add(-1.0*coefficient, pod_vectors.at(pod_vector_n));

          double error {0.0};
          for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
            {
              mass_matrix.vmult(temp, snapshot_residue.block(dim_n));
              error += temp * snapshot_residue.block(dim_n);
            }
          projection_errors.at(pod_vector_n) += error;
        }
    }

  for (auto &value : projection_errors)
    {
      std::cout << std::setprecision(50) << value << std::endl;
    }
}
