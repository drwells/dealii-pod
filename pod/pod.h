#ifndef __deal2__pod_h
#define __deal2__pod_h
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/bundled/boost/archive/text_iarchive.hpp>
// needed to get around the "save the dof handler issue"
#include <deal.II/dofs/dof_faces.h>
#include <deal.II/dofs/dof_levels.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <glob.h>
#include <iostream>
#include <memory>
#include <vector>

#include "../h5/h5.h"
#include "../extra/extra.h"

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

  template<int dim>
  class PODOutput
  {
  public:
    PODOutput();
    PODOutput(const std::shared_ptr<DoFHandler<dim>> dof_handler,
              const std::shared_ptr<BlockVector<double>> mean_vector,
              const std::shared_ptr<std::vector<BlockVector<double>>> pod_vectors,
              const std::string filename_base);
    void reinit(const std::shared_ptr<DoFHandler<dim>> dof_handler,
                const std::shared_ptr<BlockVector<double>> mean_vector,
                const std::shared_ptr<std::vector<BlockVector<double>>> pod_vectors,
                const std::string filename_base);
    void save_solution(const Vector<double> &solution,
                       const double time,
                       const unsigned int timestep_number);
  private:
    std::string filename_base;
    std::shared_ptr<const DoFHandler<dim>> scalar_dof_handler;
    std::shared_ptr<const BlockVector<double>> mean_vector;
    std::shared_ptr<const std::vector<BlockVector<double>>> pod_vectors;
    std::unique_ptr<FESystem<dim>> vector_fe;
    DoFHandler<dim> vector_dof_handler;
    std::vector<XDMFEntry> xdmf_entries;
    bool write_mesh;
  };

  template<int dim>
  PODOutput<dim>::PODOutput()
  :
  vector_fe {new FESystem<dim>(FE_Q<dim>(1), 1)}
  {}

  template<int dim>
  PODOutput<dim>::PODOutput
  (const std::shared_ptr<DoFHandler<dim>> dof_handler,
   const std::shared_ptr<BlockVector<double>> mean_vector,
   const std::shared_ptr<std::vector<BlockVector<double>>> pod_vectors,
   const std::string filename_base)
    :
    vector_fe {new FESystem<dim>(FE_Q<dim>(1), 1)}
  {
    reinit(dof_handler, mean_vector, pod_vectors, filename_base);
  }

  template<int dim>
  void PODOutput<dim>::reinit
  (const std::shared_ptr<DoFHandler<dim>> dof_handler,
   const std::shared_ptr<BlockVector<double>> mean_vector,
   const std::shared_ptr<std::vector<BlockVector<double>>> pod_vectors,
   std::string filename_base)
  {
    this->filename_base = filename_base;
    scalar_dof_handler
    = std::const_pointer_cast<const DoFHandler<dim>>(dof_handler);
    this->mean_vector
    = std::const_pointer_cast<const BlockVector<double>>(mean_vector);
    this->pod_vectors
    = std::const_pointer_cast<const std::vector<BlockVector<double>>>(pod_vectors);
    vector_fe = std::unique_ptr<FESystem<dim>>
      {new FESystem<dim>(dof_handler->get_fe(), dim)};
    write_mesh = true;
    vector_dof_handler.initialize(dof_handler->get_tria(), *vector_fe);
  }

  template<int dim>
  void PODOutput<dim>::save_solution(const Vector<double> &solution,
                                     const double time,
                                     const unsigned int timestep_number)
  {
    std::string filename = filename_base
      + extra::int_to_string(timestep_number, 15)
      + ".h5";
    const unsigned int n_dofs = mean_vector->block(0).size();
    const unsigned int n_pod_dofs = pod_vectors->size();
    BlockVector<double> fe_solution(dim, n_dofs);
    fe_solution = *mean_vector;
    for (unsigned int i = 0; i < n_pod_dofs; ++i)
      {
        fe_solution.add(solution(i), pod_vectors->at(i));
      }

    std::vector<std::string> solution_names(dim, "v");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      component_interpretation
      (dim, DataComponentInterpretation::component_is_part_of_vector);

    dealii::Vector<double> vector_solution (vector_dof_handler.n_dofs());
    std::vector<types::global_dof_index> loc_vector_dof_indices (vector_fe->dofs_per_cell),
      loc_component_dof_indices (scalar_dof_handler->get_fe().dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
      vector_cell = vector_dof_handler.begin_active(),
      vector_endc = vector_dof_handler.end(),
      component_cell = scalar_dof_handler->begin_active();
    for (; vector_cell != vector_endc; ++vector_cell, ++component_cell)
      {
        vector_cell->get_dof_indices(loc_vector_dof_indices);
        component_cell->get_dof_indices(loc_component_dof_indices);
        for (unsigned int j = 0; j < vector_fe->dofs_per_cell; ++j)
          {
            switch (vector_fe->system_to_base_index(j).first.first)
              {
                // TODO this is sloppy cut-and-paste from step-35
              case 0:
                vector_solution(loc_vector_dof_indices[j]) =
                  fe_solution.block(vector_fe->system_to_base_index(j).first.second)
                  (loc_component_dof_indices[vector_fe->system_to_base_index(j).second]);
                break;
              default:
                ExcInternalError();
              }
          }
      }
    DataOut<dim> data_out;
    data_out.attach_dof_handler(vector_dof_handler);

    data_out.add_data_vector(vector_solution, solution_names,
                             DataOut<dim>::type_dof_data,
                             component_interpretation);
    data_out.build_patches(2); // TODO unhardcode the patch level
    std::string solution_file_name = filename_base
      + extra::int_to_string(timestep_number, 10)
      + ".h5";
    std::string mesh_file_name = "mesh.h5";
    std::string xdmf_filename = "solution.xdmf";

    DataOutBase::DataOutFilter data_filter
      (DataOutBase::DataOutFilterFlags(true, true));
    data_out.write_filtered_data(data_filter);
    data_out.write_hdf5_parallel(data_filter, write_mesh, mesh_file_name,
                                 solution_file_name, MPI_COMM_WORLD);

    // only save the mesh once.
    write_mesh = false;
    auto new_xdmf_entry = data_out.create_xdmf_entry
      (data_filter, mesh_file_name, solution_file_name, time, MPI_COMM_WORLD);
    xdmf_entries.push_back(std::move(new_xdmf_entry));
    data_out.write_xdmf_file(xdmf_entries, xdmf_filename, MPI_COMM_WORLD);
  }


  template<int dim>
  void create_dof_handler_from_triangulation_file
  (const std::string &file_name,
   const bool &renumber,
   const FE_Q<dim> &fe,
   DoFHandler<dim> &dof_handler,
   Triangulation<dim> &triangulation)
  {
    std::filebuf file_buffer;
    file_buffer.open(file_name, std::ios::in);
    std::istream in_stream (&file_buffer);
    boost::archive::text_iarchive archive(in_stream);
    archive >> triangulation;
    dof_handler.initialize(triangulation, fe);
    if (renumber)
      {
        DoFRenumbering::boost::Cuthill_McKee(dof_handler);
      }
  }

  void load_pod_basis(const std::string &pod_vector_glob,
                      const std::string &mean_vector_file_name,
                      BlockVector<double> &mean_vector,
                      std::vector<BlockVector<double>> &pod_vectors);

  void method_of_snapshots(const dealii::SparseMatrix<double> &mass_matrix,
                           const std::vector<std::string> &snapshot_file_names,
                           const unsigned int n_pod_vectors,
                           const bool center_trajectory,
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
