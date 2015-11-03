#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/matrix_tools.h>

#include <boost/math/special_functions/round.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "parameters.h"

#include "../pod/pod.h"
#include "../extra/extra.h"
#include "../h5/h5.h"

using namespace dealii;

constexpr unsigned int dim {2};
constexpr double timestep_tolerance {1e-8};
int main()
{
  Parameters parameters;
  auto snapshot_file_names = extra::expand_file_names(parameters.snapshot_glob);

  FullMatrix<double> pod_coefficients;
  H5::load_full_matrix(parameters.pod_coefficients_file_name, pod_coefficients);
  const double rom_time_step {(parameters.rom_stop_time - parameters.rom_start_time)
      /(pod_coefficients.m() - 1)};

  std::vector<BlockVector<double>> pod_vectors;
  BlockVector<double> mean_vector;
  POD::load_pod_basis(parameters.pod_vector_glob, parameters.mean_vector_file_name,
                      mean_vector, pod_vectors);

  FE_Q<dim> fe(parameters.fe_order);
  QGauss<dim> quad((3*fe.degree + 2)/2);
  Triangulation<dim> triangulation;
  DoFHandler<dim> dof_handler;
  SparsityPattern sparsity_pattern;
  POD::create_dof_handler_from_triangulation_file
    ("triangulation.txt", parameters.renumber, fe, dof_handler, triangulation);

  const unsigned int n_dofs = pod_vectors.at(0).block(0).size();
  const unsigned int n_pod_dofs = pod_vectors.size();

  {
    DynamicSparsityPattern d_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, d_sparsity);
    sparsity_pattern.copy_from(d_sparsity);
  }

  SparseMatrix<double> mass_matrix(sparsity_pattern);
  MatrixCreator::create_mass_matrix(dof_handler, quad, mass_matrix);

  double error {0.0};
  Vector<double> temp(mean_vector.block(0).size());
  const double snapshot_time_step
    {(parameters.snapshot_stop_time - parameters.snapshot_start_time)
    /(snapshot_file_names.size() - 1)};
  double snapshot_current_time {parameters.snapshot_start_time};

  BlockVector<double> solution_difference;
  BlockVector<double> current_snapshot;
  for (unsigned int snapshot_n = 0; snapshot_n < snapshot_file_names.size();
       ++snapshot_n)
    {
      if (snapshot_current_time > parameters.rom_stop_time
          or snapshot_current_time < parameters.rom_start_time)
        {
        }
      else
        {
          H5::load_block_vector(snapshot_file_names.at(snapshot_n), current_snapshot);

          solution_difference = mean_vector;
          int rom_row_index = boost::math::iround
            ((snapshot_current_time - parameters.rom_start_time)/rom_time_step);
          if (rom_row_index < 0)
            {
              std::cerr << "The ROM row index must be positive." << std::endl;
            }
          Assert(rom_row_index >= 0, ExcInternalError());
          double rom_row_index_d = (snapshot_current_time - parameters.rom_start_time)
            /rom_time_step;
          if (std::abs(boost::math::iround(rom_row_index_d) - rom_row_index_d)
              > timestep_tolerance)
            {
              std::cerr << "current time: " << std::setprecision(51)
                        << snapshot_current_time
                        << std::endl;
              std::cerr << "The ROM row index ("
                        << rom_row_index_d
                        << ") must be integral." << std::endl;
              std::exit(EXIT_FAILURE);
            }

          std::cout << "C("
                    << rom_row_index
                    << ", "
                    << "0) = "
                    << pod_coefficients(rom_row_index, 0)
                    << std::endl;
          for (unsigned int pod_vector_n = 0; pod_vector_n < pod_vectors.size();
               ++pod_vector_n)
            {
              solution_difference.add(pod_coefficients(rom_row_index, pod_vector_n),
                                      pod_vectors.at(pod_vector_n));
            }
          solution_difference -= current_snapshot;

          double current_error {0.0};
          for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
            {
              mass_matrix.vmult(temp, solution_difference.block(dim_n));
              current_error += temp * solution_difference.block(dim_n);
            }
          std::cout << std::setprecision(20)
                    << std::sqrt(current_error)
                    << std::endl;
          error += std::sqrt(current_error);
        }
      snapshot_current_time += snapshot_time_step;
    }
}
