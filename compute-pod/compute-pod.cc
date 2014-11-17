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
    Triangulation<dim>                    triangulation;
    FE_Q<dim>                             fe;
    DoFHandler<dim>                       dof_handler;
    SparsityPattern                       sparsity_pattern;
    int                                   num_pod_vectors;
    std::string                           snapshot_glob;
    std::string                           mesh_file_name;
    std::map<int, dealii::Vector<double>> snapshots;
    dealii::SparseMatrix<double>          mass_matrix;
    POD::PODBasis                         pod_result;

  public:
    PODVectors (int num_pod_vectors)
      :
      fe(1),
      dof_handler()
    {
      this->num_pod_vectors = num_pod_vectors;
      snapshot_glob = "snapshot-*.txt";
      mesh_file_name = "solution-000.vtk";
    }

    void load_mesh()
    {
      dealii::GridIn<dim> grid_in;
      grid_in.attach_triangulation(triangulation);
      std::filebuf file_buffer;
      if (file_buffer.open(mesh_file_name, std::ios::in))
        {
          std::istream input_stream(&file_buffer);
          grid_in.read_vtk(input_stream);
          file_buffer.close();
        }
      else
        {
          ExcIO();
        }
      dof_handler.initialize(triangulation, fe);
      std::cout << "DoFs: " << dof_handler.n_dofs() << std::endl;
      std::cout << "active cells: " << triangulation.n_active_cells() << std::endl;

      QGauss<dim> quadrature_rule(fe.degree + 1);

      CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, c_sparsity);
      sparsity_pattern.copy_from(c_sparsity);

      mass_matrix.reinit(sparsity_pattern);
      MatrixCreator::create_mass_matrix(dof_handler, quadrature_rule, mass_matrix);
    }

    void load_snapshots()
    {
      glob_t glob_result;
      glob(snapshot_glob.c_str(), GLOB_TILDE, nullptr, &glob_result);
      std::cout << "number of snapshots: " << glob_result.gl_pathc << std::endl;
      for (size_t i = 0; i < glob_result.gl_pathc; ++i)
        {
          std::ifstream input_stream(std::string(glob_result.gl_pathv[i]));
          std::istream_iterator<double> start(input_stream), end;
// TODO there seems to be some bug with initializing a dealii::Vector from iterators in this way.
// This copy should not be necessary.
          std::vector<double> _snapshot(start, end);
          dealii::Vector<double> snapshot(_snapshot.begin(), _snapshot.end());
          snapshots.insert(std::pair<int, dealii::Vector<double>>(i, snapshot));
        }
      globfree(&glob_result);
      std::cout << "Loaded snapshots." << std::endl;
    }

    void compute_pod_basis()
    {
      POD::pod_basis(mass_matrix, snapshots, num_pod_vectors, pod_result);
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

      for (int i = 0; i < num_pod_vectors; ++i)
        {
          std::ostringstream file_name;
          file_name << "podvector-" << i << ".txt";
          std::ofstream pod_vector_stream;
          pod_vector_stream.open(file_name.str());
          for (auto value : pod_result.vectors[i])
            {
              pod_vector_stream << value << " ";
            }
          pod_vector_stream.close();
        }
    }

    void run()
    {
      load_mesh();
      load_snapshots();
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