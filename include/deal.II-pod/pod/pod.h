/* ---------------------------------------------------------------------
 * Copyright (C) 2014-2015 David Wells
 *
 * This file is NOT part of the deal.II library.
 *
 * This file is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 *
 * Author: David Wells, Virginia Tech, 2014-2015;
 *         David Wells, Rensselaer Polytechnic Institute, 2015
 */
#ifndef dealii__rom_pod_h
#define dealii__rom_pod_h
#include <deal.II/base/utilities.h>

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

#include <boost/archive/text_iarchive.hpp>

#include <memory>
#include <vector>

namespace POD
{
  using namespace dealii;

  class BlockPODBasis
  {
  public:
    BlockPODBasis();
    BlockPODBasis(unsigned int n_blocks, unsigned int n_dofs_per_block);

    void reinit(unsigned int n_blocks, unsigned int n_dofs_per_block);
    void project_load_vector(BlockVector<double> &load_vector,
                             BlockVector<double> &pod_load_vector) const;
    void project_to_fe(const BlockVector<double> &pod_vector,
                       BlockVector<double> &fe_vector) const;

    std::vector<BlockVector<double>> vectors;
    BlockVector<double> mean_vector;
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

  void load_pod_basis(const std::string                &pod_vector_glob,
                      const std::string                &mean_vector_file_name,
                      BlockVector<double>              &mean_vector,
                      std::vector<BlockVector<double>> &pod_vectors);

  void method_of_snapshots(const SparseMatrix<double>     &mass_matrix,
                           const std::vector<std::string> &snapshot_file_names,
                           const unsigned int             n_pod_vectors,
                           const bool                     center_trajectory,
                           BlockPODBasis                  &pod_basis);

  void create_reduced_matrix(const std::vector<BlockVector<double>> &pod_vectors,
                             const SparseMatrix<double>             &full_matrix,
                             FullMatrix<double>                     &rom_matrix);


  void create_reduced_matrix(const std::vector<BlockVector<double>> &pod_vectors,
                             const SparseMatrix<double>             &full_matrix,
                             const std::vector<unsigned int>        &dims,
                             FullMatrix<double>                     &rom_matrix);

  template<int dim>
  void create_dof_handler_from_triangulation_file
  (const std::string  &file_name,
   const bool         &renumber,
   const FE_Q<dim>    &fe,
   DoFHandler<dim>    &dof_handler,
   Triangulation<dim> &triangulation);
}
#endif
