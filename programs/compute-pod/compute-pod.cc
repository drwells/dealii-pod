#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

#include <deal.II-pod/extra/extra.h>
#include <deal.II-pod/pod/pod.h>
#include <deal.II-pod/h5/h5.h>

#include "parameters.h"


namespace POD
{
  using namespace dealii;

  template<int dim>
  class PODVectors
  {
  public:
    PODVectors(const Parameters &parameters);
    void run();
  private:
    const Parameters     parameters;
    Triangulation<dim>   triangulation;
    SparsityPattern      sparsity_pattern;
    const FE_Q<dim>      fe;
    DoFHandler<dim>      dof_handler;
    const QGauss<dim>    quadrature_rule;
    SparseMatrix<double> mass_matrix;
    POD::BlockPODBasis   pod_result;
    const FESystem<dim>  vector_fe;
    DoFHandler<dim>      vector_dof_handler;

    void load_mesh();
    void compute_pod_basis();
    void save_pod_basis();
  };


  template<int dim>
  PODVectors<dim>::PODVectors(const Parameters &parameters)
    :
    parameters(parameters),
    fe(parameters.fe_order),
    quadrature_rule(parameters.fe_order + 3),
    vector_fe(fe, dim)
  {}


  template<int dim>
  void PODVectors<dim>::load_mesh()
  {
    std::ifstream in_stream(parameters.triangulation_file_name);
    boost::archive::text_iarchive archive(in_stream);
    archive >> triangulation;

    dof_handler.initialize(triangulation, fe);
    vector_dof_handler.initialize(triangulation, vector_fe);

    if (parameters.renumber)
      {
        DoFRenumbering::boost::Cuthill_McKee(dof_handler);
        DoFRenumbering::boost::Cuthill_McKee(vector_dof_handler);
      }

    DynamicSparsityPattern d_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, d_sparsity);
    sparsity_pattern.copy_from(d_sparsity);
    mass_matrix.reinit(sparsity_pattern);
    MatrixCreator::create_mass_matrix(dof_handler, quadrature_rule, mass_matrix);
  }


  template<int dim>
  void PODVectors<dim>::compute_pod_basis()
  {
    auto snapshot_file_names = extra::expand_file_names(parameters.snapshot_glob);

    method_of_snapshots(mass_matrix, snapshot_file_names, parameters.n_pod_vectors,
                        parameters.center_trajectory, pod_result);
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

        if (parameters.save_plot_pictures)
          {
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
            std::vector<types::global_dof_index> loc_vector_dof_indices
              (vector_fe.dofs_per_cell), loc_component_dof_indices(fe.dofs_per_cell);
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
                          pod_result.vectors[i].block
                          (vector_fe.system_to_base_index(j).first.second)
                          (loc_component_dof_indices[vector_fe.system_to_base_index(j).second]);
                        break;
                      default:
                        ExcInternalError();
                      }
                  }
              }

            data_out.add_data_vector
              (vector_solution, solution_names, DataOut<dim>::type_dof_data,
               component_interpretation);
            data_out.build_patches (2);
            data_out.write_filtered_data(data_filter);

            data_out.write_hdf5_parallel
              (data_filter, write_mesh, mesh_file_name,
                                         plot_file_name, MPI_COMM_WORLD);
            write_mesh = false;

            auto new_xdmf_entry = data_out.create_xdmf_entry
              (data_filter, mesh_file_name, plot_file_name,
               static_cast<double>(i), MPI_COMM_WORLD);
            xdmf_entries.push_back(std::move(new_xdmf_entry));
            data_out.write_xdmf_file(xdmf_entries, xdmf_filename, MPI_COMM_WORLD);
          }
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
    POD::Parameters parameters;
    parameters.read_data("parameter-file.prm");
    if (parameters.dimension == 2)
      {
        PODVectors<2> pod_vectors(parameters);
        pod_vectors.run();
      }
    else
      {
        PODVectors<3> pod_vectors(parameters);
        pod_vectors.run();
      }
  }
}
