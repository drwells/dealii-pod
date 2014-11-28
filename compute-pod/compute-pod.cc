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

#include <deal.II/bundled/boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <glob.h>

#include "../pod/pod.h"
#include "../h5/h5.h"

namespace POD
{
  using namespace dealii;

  template<int dim>
  class PODVectors
  {
  public:
    PODVectors(int n_pod_vectors);
    void run();
  private:
    FE_Q<dim>                    fe;
    unsigned int                 n_pod_vectors;
    std::string                  snapshot_glob;
    std::string                  triangulation_file_name;
    SparsityPattern              sparsity_pattern;
    dealii::SparseMatrix<double> mass_matrix;
    POD::BlockPODBasis           pod_result;
    void load_mesh();
    void compute_pod_basis();
    void save_pod_basis();
  };

  template<int dim>
  PODVectors<dim>::PODVectors (int n_pod_vectors)
    :
    fe(2), // TODO this depends on the parameter file.
    pod_result()
  {
    this->n_pod_vectors = n_pod_vectors;
    snapshot_glob = "snapshot-*.h5";
    triangulation_file_name = "triangulation.txt";
  }

  template<int dim>
  void PODVectors<dim>::load_mesh()
  {
    // TODO is it possible to load the DoFHandler?
    Triangulation<dim> triangulation;
    std::filebuf file_buffer;
    file_buffer.open (triangulation_file_name.c_str (), std::ios::in);
    std::istream in_stream (&file_buffer);
    boost::archive::text_iarchive archive(in_stream);
    archive >> triangulation;

    DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    std::cout << "DoFs: " << dof_handler.n_dofs() << std::endl;
    std::cout << "active cells: " << triangulation.n_active_cells() << std::endl;

    QGauss<dim> quadrature_rule(fe.degree + 1);

    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, c_sparsity);
    sparsity_pattern.copy_from(c_sparsity);
    mass_matrix.reinit(sparsity_pattern);
    MatrixCreator::create_mass_matrix(dof_handler, quadrature_rule, mass_matrix);
  }

  template<int dim>
  void PODVectors<dim>::compute_pod_basis()
  {
    std::vector<std::string> snapshot_file_names;

    glob_t glob_result;
    glob(snapshot_glob.c_str (), GLOB_TILDE, nullptr, &glob_result);
    for (unsigned int i = 0; i < glob_result.gl_pathc; ++i)
      {
        snapshot_file_names.push_back (std::string (glob_result.gl_pathv[i]));
      }
    globfree(&glob_result);

    method_of_snapshots(mass_matrix, snapshot_file_names, n_pod_vectors,
                        pod_result);
  }

  template<int dim>
  void PODVectors<dim>::save_pod_basis()
  {
    std::ofstream singular_values_stream;
    singular_values_stream.open("singular_values.txt");
    for (auto singular_value : pod_result.singular_values)
      {
        singular_values_stream << singular_value << std::endl;
      }

    for (int i = 0; i < n_pod_vectors; ++i)
      {
        std::string file_name = "pod-vector-" + Utilities::int_to_string(i, 5)
          + ".h5";
        H5::save_block_vector(file_name, pod_result.vectors[i]);
      }
  }

  template<int dim>
  void PODVectors<dim>::run()
  {
    load_mesh();
    compute_pod_basis();
    save_pod_basis();
  }
}

int main(int argc, char **argv)
{
  using namespace POD;
  PODVectors<2> pod_vectors(3);
  pod_vectors.run();
}
