#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/full_matrix.h>
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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

#include "../ns/ns.h"
#include "../extra/extra.h"
#include "../pod/pod.h"
#include "../h5/h5.h"
#include "parameters.h"


namespace ComputePOD
{
  using namespace dealii;

  template<int dim>
  class ComputePODMatrices
  {
  public:
    ComputePODMatrices(const Parameters &parameters);
    void run();
  private:
    void load_pod_vectors();

    void setup_mass_matrix();
    void setup_laplace_matrix();
    void setup_boundary_matrix();
    void setup_advective_linearization_matrix();
    void setup_gradient_linearization_matrix();
    void setup_nonlinearity();

    void save_rom_components();

    const Parameters parameters;

    FE_Q<dim> fe;
    QGauss<dim> quad;
    Triangulation<dim> triangulation;
    SparsityPattern sparsity_pattern;
    DoFHandler<dim> dof_handler;

    std::vector<BlockVector<double>> pod_vectors;
    BlockVector<double> mean_vector;
    unsigned int n_dofs;
    unsigned int n_pod_dofs;

    FullMatrix<double> mass_matrix;
    FullMatrix<double> laplace_matrix;
    FullMatrix<double> boundary_matrix;

    FullMatrix<double> gradient_matrix;
    FullMatrix<double> advection_matrix;

    std::vector<FullMatrix<double>> nonlinearity;

    Vector<double> mean_contribution;
    Vector<double> initial;
  };

  template<int dim>
  ComputePODMatrices<dim>::ComputePODMatrices
  (const Parameters &params)
    :
    parameters (params),
    fe(params.fe_order),
    quad(params.fe_order + 2)
  {
    POD::create_dof_handler_from_triangulation_file
    ("triangulation.txt", parameters.renumber, fe, dof_handler, triangulation);

    {
      DynamicSparsityPattern d_sparsity(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, d_sparsity);
      sparsity_pattern.copy_from(d_sparsity);
    }
  }



  template<int dim>
  void
  ComputePODMatrices<dim>::load_pod_vectors()
  {
    POD::load_pod_basis("pod-vector*.h5", "mean-vector.h5", mean_vector, pod_vectors);
    n_dofs = pod_vectors.at(0).block(0).size();
    n_pod_dofs = pod_vectors.size();

    mean_contribution.reinit(n_pod_dofs);
  }



  template<int dim>
  void
  ComputePODMatrices<dim>::setup_mass_matrix()
  {
    // This function is a slight misnomer: I also project the initial
    // condition here too.
    SparseMatrix<double> full_mass_matrix(sparsity_pattern);
    MatrixCreator::create_mass_matrix(dof_handler, quad, full_mass_matrix);
    POD::create_reduced_matrix(pod_vectors, full_mass_matrix, mass_matrix);

    BlockVector<double> centered_initial;
    H5::load_block_vector("initial.h5", centered_initial);
    initial.reinit(n_pod_dofs);
    centered_initial -= mean_vector;
    for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
      {
        Vector<double> temp(n_dofs);
        full_mass_matrix.vmult(temp, centered_initial.block(dim_n));
        for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs;
             ++pod_vector_n)
          {
            initial[pod_vector_n] +=
              temp * pod_vectors.at(pod_vector_n).block(dim_n);
          }
      }
  }



  template<int dim>
  void
  ComputePODMatrices<dim>::setup_laplace_matrix()
  {
    // also a misnomer: this sets up the Laplace matrix *and* subtracts off
    // the relevant part from the mean contribution.
    SparseMatrix<double> full_laplace_matrix(sparsity_pattern);
    MatrixCreator::create_laplace_matrix(dof_handler, quad, full_laplace_matrix);
    POD::create_reduced_matrix(pod_vectors, full_laplace_matrix, laplace_matrix);

    for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
      {
        Vector<double> temp(n_dofs);
        full_laplace_matrix.vmult(temp, mean_vector.block(dim_n));
        for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs; ++pod_vector_n)
          {
            mean_contribution(pod_vector_n) -= 1.0/parameters.reynolds_n*
              (temp * pod_vectors.at(pod_vector_n).block(dim_n));
          }
      }
  }



  template<int dim>
  void
  ComputePODMatrices<dim>::setup_boundary_matrix()
  {
    // Same as above: setup the matrix and subtract off the relevant piece of
    // the mean contribution vector.
    SparseMatrix<double> full_boundary_matrix(sparsity_pattern);
    QGauss<dim - 1> face_quad(fe.degree + 2);
    POD::NavierStokes::create_boundary_matrix
      (dof_handler, face_quad, parameters.outflow_label, full_boundary_matrix);

    std::vector<unsigned int> dims {0};
    POD::create_reduced_matrix(pod_vectors, full_boundary_matrix, dims,
                               boundary_matrix);

    Vector<double> temp(n_dofs);
    full_boundary_matrix.vmult(temp, mean_vector.block(0));
    for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs; ++pod_vector_n)
      {
        mean_contribution(pod_vector_n) += 1.0/parameters.reynolds_n
          *(temp * pod_vectors.at(pod_vector_n).block(0));
      }
  }



  template<int dim>
  void
  ComputePODMatrices<dim>::setup_advective_linearization_matrix()
  {
    QGauss<dim> higher_quadrature(2*(parameters.fe_order + 1));
    POD::NavierStokes::create_reduced_advective_linearization
    (dof_handler, sparsity_pattern, higher_quadrature, mean_vector, pod_vectors,
     advection_matrix);
  }



  template<int dim>
  void
  ComputePODMatrices<dim>::setup_gradient_linearization_matrix()
  {
    QGauss<dim> higher_quadrature(2*(parameters.fe_order + 1));
    POD::NavierStokes::create_reduced_gradient_linearization
    (dof_handler, sparsity_pattern, higher_quadrature, mean_vector, pod_vectors,
     pod_vectors, gradient_matrix);
  }



  template<int dim>
  void
  ComputePODMatrices<dim>::setup_nonlinearity()
  {
    QGauss<dim> higher_quadrature(2*(parameters.fe_order + 1));

    Vector<double> nonlinear_contribution(n_pod_dofs);
    POD::NavierStokes::create_nonlinear_centered_contribution
      (dof_handler, sparsity_pattern, higher_quadrature, mean_vector,
       mean_vector, pod_vectors, nonlinear_contribution);
    mean_contribution.add(-1.0, nonlinear_contribution);

    POD::NavierStokes::create_reduced_nonlinearity
    (dof_handler, sparsity_pattern, higher_quadrature, pod_vectors, pod_vectors,
     nonlinearity);
  }



  template<int dim>
  void
  ComputePODMatrices<dim>::save_rom_components()
  {
    H5::save_full_matrix("rom-mass-matrix.h5", mass_matrix);
    std::cout << "saved the mass matrix." << std::endl;
    H5::save_full_matrix("rom-laplace-matrix.h5", laplace_matrix);
    std::cout << "saved the laplace matrix." << std::endl;
    H5::save_full_matrix("rom-boundary-matrix.h5", boundary_matrix);
    std::cout << "saved the boundary matrix." << std::endl;
    H5::save_full_matrix("rom-gradient-matrix.h5", gradient_matrix);
    std::cout << "saved the gradient matrix." << std::endl;
    H5::save_full_matrix("rom-advection-matrix.h5", advection_matrix);
    std::cout << "saved the advection matrix." << std::endl;
    H5::save_vector("rom-mean-contribution.h5", mean_contribution);
    std::cout << "saved the mean contribution." << std::endl;
    H5::save_full_matrices("rom-nonlinearity.h5", nonlinearity);
    std::cout << "saved the nonlinearity." << std::endl;
  }



  template<int dim>
  void
  ComputePODMatrices<dim>::run()
  {
    load_pod_vectors();
    std::cout << "loaded the POD basis." << std::endl;
    setup_mass_matrix();
    std::cout << "set up the mass matrix." << std::endl;
    setup_laplace_matrix();
    std::cout << "set up the laplace matrix." << std::endl;
    setup_boundary_matrix();
    std::cout << "set up the boundary matrix." << std::endl;
    setup_advective_linearization_matrix();
    std::cout << "set up the advection matrix." << std::endl;
    setup_gradient_linearization_matrix();
    std::cout << "set up the gradient matrix." << std::endl;
    setup_nonlinearity();
    std::cout << "set up the nonlinearity matrices." << std::endl;
    save_rom_components();
    std::cout << "saved everything to disk." << std::endl;
  }
}




int main(int argc, char **argv)
{
  using namespace POD;
  Utilities::MPI::MPI_InitFinalize mpi_initialization
  (argc, argv, numbers::invalid_unsigned_int);
  {
    ComputePOD::Parameters parameters;
    parameters.read_data("parameter-file.prm");
    ComputePOD::ComputePODMatrices<3> pod_matrices(parameters);
    pod_matrices.run();
  }
}
