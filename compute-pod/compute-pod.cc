#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/matrix_tools.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <glob.h>

#include "pod.h"

namespace POD
{
  using namespace dealii;

  template<int dim>
  class PODVectors
  {
  private:
    FE_Q<dim>                    fe;
    DoFHandler<dim>              dof_handler;
    SparsityPattern              sparsity_pattern;
    unsigned int                 n_pod_vectors;
    std::string                  snapshot_glob;
    std::string                  mesh_file_name;
    dealii::SparseMatrix<double> mass_matrix;
    POD::BlockPODBasis           pod_result;

  public:
    PODVectors (int n_pod_vectors)
      :
      fe(1),
      dof_handler()
    {
      this->n_pod_vectors = n_pod_vectors;
      snapshot_glob = "snapshot-*.h5";
      mesh_file_name = "solution-000.vtk";
    }

    void load_mesh()
    {
      // TODO load the DoFHandler
      std::cout << "DoFs: " << dof_handler.n_dofs() << std::endl;
      std::cout << "active cells: " << triangulation.n_active_cells() << std::endl;

      QGauss<dim> quadrature_rule(fe.degree + 1);

      CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, c_sparsity);
      sparsity_pattern.copy_from(c_sparsity);
      mass_matrix.reinit(sparsity_pattern);
      MatrixCreator::create_mass_matrix(dof_handler, quadrature_rule, mass_matrix);
    }

    void compute_pod_basis()
    {
      std::vector<std::string> snapshot_file_names;
      std::string dataset_name("/v");
      method_of_snapshots(mass_matrix, snapshot_file_names, dataset_name,
                          n_pod_vectors, pod_basis);
    }

    void save_pod_basis()
    {
      std::ofstream singular_values_stream;
      singular_values_stream.open("singular_values.txt");
      for (auto singular_value : pod_result.singular_values)
        {
          singular_values_stream << singular_value << std::endl;
        }
      singular_values_stream.close();

      // for (int i = 0; i < n_pod_vectors; ++i)
      //   {
      //     std::ostringstream file_name;
      //     file_name << "podvector-" << i << ".txt";
      //     std::ofstream pod_vector_stream;
      //     pod_vector_stream.open(file_name.str());
      //     for (auto value : pod_result.vectors[i])
      //       {
      //         pod_vector_stream << value << " ";
      //       }
      //     pod_vector_stream.close();
      //   }
    }

    void run()
    {
      load_mesh();
      compute_pod_basis();
      save_pod_basis();
    }
  };
}

int main(int argc, char **argv)
{
  using namespace POD;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  {
    PODVectors<2> pod_vectors(10);
    pod_vectors.run();
  }
}
