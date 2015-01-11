#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/bundled/boost/archive/text_iarchive.hpp>
// These two are needed to get around issue 278; see
// https://github.com/dealii/dealii/pull/278
#include <deal.II/dofs/dof_faces.h>
#include <deal.II/dofs/dof_levels.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <glob.h>

#include "../pod/pod.h"
#include "../h5/h5.h"


namespace POD
{
  using namespace dealii;

  constexpr unsigned int deg = 2;

  template<int dim>
  class PODVectors
  {
  public:
    PODVectors(bool renumber);
    void run();
  private:
    bool                           renumber;
    unsigned int                   n_pod_vectors;
    std::string                    snapshot_glob;
    std::string                    triangulation_file_name;
    SparsityPattern                sparsity_pattern;
    dealii::SparseMatrix<double>   mass_matrix;
    POD::BlockPODBasis             pod_result;
    Triangulation<dim>             triangulation;
    FE_Q<dim>                      fe;
    DoFHandler<dim>                dof_handler;
    FESystem<dim>                  vector_fe;
    DoFHandler<dim>                vector_dof_handler;
    QGauss<dim>                    quadrature_rule;


    void load_mesh();
    void compute_pod_basis();
    void save_pod_basis();
  };


  template<int dim>
  PODVectors<dim>::PODVectors (bool renumber)
    :
    n_pod_vectors {100},
    fe(deg), // TODO this depends on the parameter file.
    vector_fe(fe, dim),
    quadrature_rule(deg + 3)
  {
    this->renumber = renumber;
    snapshot_glob = "snapshot-*.h5";
    triangulation_file_name = "triangulation.txt";
  }


  template<int dim>
  void PODVectors<dim>::load_mesh()
  {
    std::filebuf file_buffer;
    file_buffer.open (triangulation_file_name, std::ios::in);
    std::istream in_stream (&file_buffer);
    boost::archive::text_iarchive archive(in_stream);
    archive >> triangulation;

    dof_handler.initialize(triangulation, fe);
    vector_dof_handler.initialize(triangulation, vector_fe);

    if (renumber)
      {
        std::cout << "renumbering." << std::endl;
        DoFRenumbering::boost::Cuthill_McKee(dof_handler);
        DoFRenumbering::boost::Cuthill_McKee(vector_dof_handler);
      }

    std::cout << "DoFs: " << dof_handler.n_dofs() << std::endl;

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

    method_of_snapshots(mass_matrix, snapshot_file_names, n_pod_vectors, pod_result);
    // check orthogonality.
    dealii::BlockVector<double> temp(pod_result.vectors.at(0).n_blocks(),
                                     pod_result.vectors.at(0).block(0).size());
    for (unsigned int i = 0; i < pod_result.get_n_pod_vectors(); ++i)
      {
        auto &right_vector = pod_result.vectors.at(i);
        for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
          {
            mass_matrix.vmult(temp.block(dim_n), right_vector.block(dim_n));
          }
        for (unsigned int j = 0; j < pod_result.get_n_pod_vectors(); ++j)
          {
            auto &left_vector = pod_result.vectors.at(j);
            double result = left_vector * temp;
            constexpr double tolerance = 1e-12;
            if (i == j)
              {
                if (std::abs(result - 1.0) > tolerance)
                  {
                    std::cerr << "C(" << j << ", " << i << ") = "
                              << result << std::endl;
                    Assert(std::abs(result - 1) < tolerance, ExcInternalError());
                  }
              }
            else
              {
                if (std::abs(result) > tolerance)
                  {
                    std::cerr << "C(" << i << ", " << j << ") = "
                              << result << std::endl;
                    Assert(std::abs(result) < tolerance, ExcInternalError());
                  }
              }
          }
      }
    std::cout << "computed the POD basis." << std::endl;
  }


  template<int dim>
  void PODVectors<dim>::save_pod_basis()
  {
    std::ofstream singular_values_stream;
    singular_values_stream.open("singular_values.txt");
    for (auto singular_value : pod_result.singular_values)
      {
        double output = std::isnan(singular_value) ? -0.0 : singular_value;
        singular_values_stream << output << std::endl;
      }
    std::string mean_vector_name = "mean-vector.h5";
    H5::save_block_vector(mean_vector_name, pod_result.mean_vector);

    std::string mesh_file_name = "mesh.h5";
    std::string xdmf_filename = "pod-vectors.xdmf";

    bool write_mesh = true;
    std::vector<XDMFEntry> xdmf_entries;

    for (unsigned int i = 0; i < pod_result.get_n_pod_vectors(); ++i)
      {
        std::string file_name = "pod-vector-" + Utilities::int_to_string(i, 7)
                                + ".h5";
        H5::save_block_vector(file_name, pod_result.vectors.at(i));

        // save the information in a plot-friendly format.
        std::vector<std::string> solution_names(dim, "v");
        std::string plot_file_name = "pod-vector-plot-"
                                     + Utilities::int_to_string(i, 7) + ".h5";

        DataOut<dim> data_out;
        data_out.attach_dof_handler(vector_dof_handler);
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        component_interpretation
        (dim,
         DataComponentInterpretation::component_is_part_of_vector);
        DataOutBase::DataOutFilter data_filter
        (DataOutBase::DataOutFilterFlags(true, true));

        dealii::Vector<double> vector_solution (vector_dof_handler.n_dofs());
        std::vector<types::global_dof_index> loc_vector_dof_indices (vector_fe.dofs_per_cell),
            loc_component_dof_indices (fe.dofs_per_cell);
        typename DoFHandler<dim>::active_cell_iterator
        vector_cell = vector_dof_handler.begin_active(),
        vector_endc = vector_dof_handler.end(),
        component_cell = dof_handler.begin_active();
        for (; vector_cell != vector_endc; ++vector_cell, ++component_cell)
          {
            vector_cell->get_dof_indices(loc_vector_dof_indices);
            component_cell->get_dof_indices(loc_component_dof_indices);
            for (unsigned int j = 0; j < vector_fe.dofs_per_cell; ++j)
              {
                switch (vector_fe.system_to_base_index(j).first.first)
                  {
                  // TODO this is sloppy cut-and-paste from step-35
                  case 0:
                    vector_solution(loc_vector_dof_indices[j]) =
                      pod_result.vectors[i].block(vector_fe.system_to_base_index(j).first.second)
                      (loc_component_dof_indices[vector_fe.system_to_base_index(j).second]);
                    break;
                  default:
                    ExcInternalError();
                  }
              }
          }

        data_out.add_data_vector (vector_solution,
                                  solution_names,
                                  DataOut<dim>::type_dof_data,
                                  component_interpretation);
        data_out.build_patches (2);
        data_out.write_filtered_data(data_filter);

        data_out.write_hdf5_parallel(data_filter, write_mesh, mesh_file_name,
                                     plot_file_name, MPI_COMM_WORLD);
        write_mesh = false;

        auto new_xdmf_entry = data_out.create_xdmf_entry
                              (data_filter, mesh_file_name, plot_file_name,
                               static_cast<double>(i), MPI_COMM_WORLD);
        xdmf_entries.push_back(std::move(new_xdmf_entry));
        data_out.write_xdmf_file(xdmf_entries, xdmf_filename, MPI_COMM_WORLD);
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
  Utilities::MPI::MPI_InitFinalize mpi_initialization
  (argc, argv, numbers::invalid_unsigned_int);
  {
    PODVectors<2> pod_vectors(false);
    pod_vectors.run();
  }
}
