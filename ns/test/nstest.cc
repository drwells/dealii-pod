#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_base.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/matrix_tools.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "../ns.h"

using namespace dealii;
namespace POD
{
  namespace NavierStokes
  {
    template<int dim>
    class Chebyshev
    {
    public:
      Chebyshev(const Tensor<1, dim, int> od) : orders(od) {}

      double operator()(const Point<dim> point) const
      {
        double result = 1.0;
        for (unsigned int i = 0; i < dim; ++i)
          {
            result *= std::cos(orders[i]*std::acos(point[i]));
          }
        return result;
      }
    private:
      Tensor<1, dim, int> orders;
    };


    template<int dim>
    void get_chebyshev_vector(const Tensor<1, dim, int> orders,
                              const DoFHandler<dim> &dof_handler,
                              const FEValues<dim>   &fe_values,
                              Vector<double>        &coefficients)
    {
      coefficients.reinit(dof_handler.n_dofs());
      auto &mapping = fe_values.get_mapping();

      Chebyshev<dim> basis_function(orders);
      std::vector<Point<dim>> support_points(dof_handler.n_dofs());
      DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);
      unsigned int i = 0;
      for (auto &point : support_points)
        {
          coefficients[i] = basis_function(point);
          ++i;
        }
    }
  }
}


using namespace POD::NavierStokes;
template<int dim>
void test_boundary_matrix(const DoFHandler<dim> &dof_handler,
                          const SparsityPattern &sparsity_pattern,
                          const unsigned int outflow_label,
                          const std::vector<BlockVector<double>> &pod_vectors)
{
  QGauss<dim - 1> face_quad(5);
  SparseMatrix<double> boundary_matrix(sparsity_pattern);
  create_boundary_matrix(dof_handler, face_quad, outflow_label, boundary_matrix);

  Vector<double> temp(pod_vectors[0].block(0).size());
  boundary_matrix.vmult(temp, pod_vectors[1].block(0));
  std::cout << "boundary integral value: "
            << pod_vectors.at(0).block(0) * temp
            << std::endl;
}


template<int dim>
void test_nonlinearity(const DoFHandler<dim> &dof_handler,
                       const std::vector<BlockVector<double>> &pod_vectors)
{
  QGauss<dim> quad (5);
  double result = trilinearity_term
  (quad, dof_handler, pod_vectors.at(0), pod_vectors.at(1), pod_vectors.at(2));
  std::cout << "nonlinearity entry is " << result << std::endl;
  std::cout << "expected value is     " << -1.488498996 << std::endl;

  result = trilinearity_term(quad, dof_handler, pod_vectors.at(0),
                             pod_vectors.at(0), pod_vectors.at(0));
  std::cout << "nonlinearity entry is " << result << std::endl;
  std::cout << "expected value is     " << -0.13799533 << std::endl;
}


template<int dim>
void test_advective_linearization
(const DoFHandler<dim> &dof_handler,
 const SparsityPattern &sparsity_pattern,
 const std::vector<BlockVector<double>> &pod_vectors)
{
  QGauss<dim> quad (5);
  SparseMatrix<double> advection_matrix(sparsity_pattern);
  create_advective_linearization(dof_handler, quad, pod_vectors.at(0),
                                 advection_matrix);
  double result = 0.0;
  Vector<double> temp(pod_vectors.at(0).block(0).size());
  for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
    {
      advection_matrix.vmult(temp, pod_vectors[0].block(dim_n));
      result += pod_vectors.at(0).block(dim_n) * temp;
    }
  std::cout << "advection integral value is " << result << std::endl;
  std::cout << "expected value is           " << -0.13799533 << std::endl;
}


template<int dim>
void test_reduced_advective_linearization
(const DoFHandler<dim>                  &dof_handler,
 const SparsityPattern                  &sparsity_pattern,
 const std::vector<BlockVector<double>> &pod_vectors)
{
  QGauss<dim> quad (5);
  const unsigned int n_pod_vectors = pod_vectors.size();

  FullMatrix<double> advection;
  create_reduced_advective_linearization
  (dof_handler, sparsity_pattern, quad, pod_vectors.at(0),
   pod_vectors, advection);

  FullMatrix<double> advection2(n_pod_vectors, n_pod_vectors);
  for (unsigned int i = 0; i < n_pod_vectors; ++i)
    {
      for (unsigned int j = 0; j < n_pod_vectors; ++j)
        {
          advection2(i, j) = trilinearity_term(quad, dof_handler,
                                               pod_vectors.at(i),
                                               pod_vectors.at(0),
                                               pod_vectors.at(j));
        }
    }
  advection.add(-1.0, advection2);
  std::cout << "reduced advection error is " << advection.l1_norm() << std::endl;
}


template<int dim>
void test_reduced_gradient_linearization
(const DoFHandler<dim>                  &dof_handler,
 const SparsityPattern                  &sparsity_pattern,
 const std::vector<BlockVector<double>> &pod_vectors)
{
  QGauss<dim> quad (5);
  const unsigned int n_pod_vectors = pod_vectors.size();

  FullMatrix<double> gradient;
  create_reduced_gradient_linearization
  (dof_handler, sparsity_pattern, quad, pod_vectors.at(0),
   pod_vectors, gradient);

  FullMatrix<double> gradient2(n_pod_vectors, n_pod_vectors);
  for (unsigned int i = 0; i < n_pod_vectors; ++i)
    {
      for (unsigned int j = 0; j < n_pod_vectors; ++j)
        {
          gradient2(i, j) = trilinearity_term(quad, dof_handler,
                                              pod_vectors.at(i),
                                              pod_vectors.at(j),
                                              pod_vectors.at(0));
        }
    }
  gradient.add(-1.0, gradient2);
  std::cout << "reduced gradient error is " << gradient.l1_norm() << std::endl;
}


template<int dim>
void test_reduced_nonlinearity
(const DoFHandler<dim>                  &dof_handler,
 const SparsityPattern                  &sparsity_pattern,
 const std::vector<BlockVector<double>> &pod_vectors)
{
  QGauss<dim> quad(5);
  std::vector<FullMatrix<double>> nonlinear_operator;
  create_reduced_nonlinearity(dof_handler, sparsity_pattern, quad,
                              pod_vectors, nonlinear_operator);

  FullMatrix<double> linearization(pod_vectors.size());
  for (unsigned int i = 0; i < pod_vectors.size(); ++i)
    {
      for (unsigned int j = 0; j < pod_vectors.size(); ++j)
        {
          for (unsigned int k = 0; k < pod_vectors.size(); ++k)
            {
              linearization(j, k) =
                trilinearity_term(quad, dof_handler, pod_vectors.at(i),
                                  pod_vectors.at(j), pod_vectors.at(k));
            }
        }
      linearization.add(-1.0, nonlinear_operator.at(i));
      std::cout << "nonlinear error on hyper row " << i << " :"
                << linearization.l1_norm() << std::endl;
    }
}


template<int dim>
void test_gradient_linearization
(DoFHandler<dim> &dof_handler,
 SparsityPattern &sparsity_pattern,
 const std::vector<BlockVector<double>> &pod_vectors)
{
  QGauss<dim> quad (5);
  ArrayArray<dim> gradient_matrices;
  for (auto &row : gradient_matrices)
    {
      for (auto &matrix : row)
        {
          matrix.reinit(sparsity_pattern);
        }
    }
  create_gradient_linearization(dof_handler, quad, pod_vectors.at(2),
                                gradient_matrices);
  BlockVector<double> temp(pod_vectors.at(0).n_blocks(),
                           pod_vectors.at(0).block(0).size());

  auto &lhs_vector = pod_vectors.at(0);
  auto &rhs_vector = pod_vectors.at(1);
  for (unsigned int row_n = 0; row_n < gradient_matrices.size(); ++row_n)
    {
      for (unsigned int column_n = 0; column_n < gradient_matrices[0].size();
           ++column_n)
        {
          gradient_matrices[row_n][column_n].vmult_add
          (temp.block(row_n), rhs_vector.block(column_n));
        }
    }
  double result = lhs_vector * temp;
  std::cout << "gradient integral value is " << result << std::endl;
  std::cout << "expected value is          " << -1.48871 << std::endl;
}


constexpr size_t dim = 2;
int main(int argc, char **argv)
{
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global (3);
  FE_Q<dim> fe (4);
  DoFHandler<dim> dof_handler (triangulation);
  QGauss<dim> quad (5);
  FEValues<dim> fe_values(fe, quad, update_values | update_gradients
                          | update_JxW_values);

  dof_handler.distribute_dofs(fe);
  std::cout << "DoFs: " << dof_handler.n_dofs() << std::endl;
  const unsigned int outflow_label = 1;
  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell != endc; ++cell)
    {
      for (unsigned int face_n = 0; face_n < GeometryInfo<dim>::faces_per_cell; ++face_n)
        {
          if (cell->face(face_n)->at_boundary())
            {
              if (std::abs(cell->face(face_n)->center()[0] - 1.0) < 1.0e-10)
                {
                  cell->face(face_n)->set_boundary_indicator(outflow_label);
                }
            }
        }
    }

  // set up the fake POD basis.
  Tensor<1, dim, int> orders;
  unsigned int n_vectors = 5;
  std::vector<BlockVector<double>> pod_vectors;
  for (unsigned int i = 0; i < n_vectors; ++i)
    {
      for (unsigned int j = 0; j < dim; ++j)
        {
          orders[j] = 3*j + 1 + 2*i;
        }
      BlockVector<double> pod_vector(dim);
      get_pod_vector(orders, dof_handler, fe_values, pod_vector.block(0));
      for (unsigned int j = 1; j < dim; ++j)
        {
          pod_vector.block(j) = pod_vector.block(0);
        }
      pod_vector.collect_sizes();
      pod_vectors.push_back(std::move(pod_vector));
      std::cout << "function order = " << orders << std::endl;
    }

  SparsityPattern sparsity_pattern;
  {
    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, c_sparsity);
    sparsity_pattern.copy_from(c_sparsity);
  }

  test_nonlinearity(dof_handler, pod_vectors);
  test_boundary_matrix(dof_handler, sparsity_pattern, outflow_label, pod_vectors);

  test_advective_linearization(dof_handler, sparsity_pattern, pod_vectors);
  test_gradient_linearization(dof_handler, sparsity_pattern, pod_vectors);

  test_reduced_advective_linearization(dof_handler, sparsity_pattern, pod_vectors);
  test_reduced_gradient_linearization(dof_handler, sparsity_pattern, pod_vectors);
  test_reduced_nonlinearity(dof_handler, sparsity_pattern, pod_vectors);
}
