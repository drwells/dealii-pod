#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_base.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/matrix_tools.h>

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
      DoFTools::map_dofs_to_support_points(mapping,
                                           dof_handler,
                                           support_points);
      unsigned int i = 0;
      for (auto &point : support_points)
        {
          coefficients[i] = basis_function(point);
          ++i;
        }
    }
  }
}

constexpr unsigned int dim = 2;

using namespace POD::NavierStokes;
int main(int argc, char **argv)
{
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global (3);
  DoFHandler<dim> dof_handler (triangulation);
  Tensor<1, dim, int> orders;
  FE_Q<dim> fe (4);
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

  // test the trilinearity.
  unsigned int n_vectors = 3;
  std::vector<BlockVector<double>> chebyshev_vectors;
  for (unsigned int i = 0; i < n_vectors; ++i)
    {
      for (unsigned int j = 0; j < dim; ++j)
        {
          orders[j] = 3*j + 1 + 2*i;
        }
      BlockVector<double> chebyshev_vector(2);
      get_chebyshev_vector(orders, dof_handler, fe_values, chebyshev_vector.block(0));
      chebyshev_vector.block(1) = chebyshev_vector.block(0);
      chebyshev_vector.collect_sizes();
      chebyshev_vectors.push_back(std::move(chebyshev_vector));
      std::cout << "function order = " << orders << std::endl;
    }

  double result = trilinearity_term(quad,
                                    dof_handler,
                                    chebyshev_vectors[0],
                                    chebyshev_vectors[1],
                                    chebyshev_vectors[2]);

  std::cout << "nonlinearity entry is " << result << std::endl;

  // test the boundary matrix.
  QGauss<dim - 1> face_quad(5);
  CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, c_sparsity);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(c_sparsity);

  SparseMatrix<double> boundary_matrix(sparsity_pattern);
  SparseMatrix<double> mass_matrix(sparsity_pattern);
  // MatrixTools::create_laplace_matrix(dof_handler, quad, mass_matrix);
  create_boundary_matrix(dof_handler, face_quad, outflow_label, boundary_matrix);

  Vector<double> temp(chebyshev_vectors[0].block(0).size());
  boundary_matrix.vmult(temp, chebyshev_vectors[1].block(0));
  std::cout << "boundary integral value: "
            << chebyshev_vectors[0].block(0) * temp
            << std::endl;

  // to destroy the pointer from dof_handler to fe:
  dof_handler.clear();
}
