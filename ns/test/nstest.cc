#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_base.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

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
  triangulation.refine_global (6);
  DoFHandler<dim> dof_handler (triangulation);
  Tensor<1, dim, int> orders;
  FE_Q<dim> fe (3); // TODO this crashes > 1
  QGauss<dim> quadrature (6);
  FEValues<dim> fe_values(fe, quadrature, update_values | update_gradients 
  | update_JxW_values);
  
  dof_handler.distribute_dofs(fe);
  std::cout << "DoFs: " << dof_handler.n_dofs() << std::endl;

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
      //chebyshev_vector.block(1) = 0.0;
      chebyshev_vector.collect_sizes();
      chebyshev_vectors.push_back(std::move(chebyshev_vector));
      std::cout << "function order = " << orders << std::endl;
    }

  double result = trilinearity_term(fe,
                                    quadrature,
                                    dof_handler,
                                    chebyshev_vectors[0],
                                    chebyshev_vectors[1],
                                    chebyshev_vectors[2]);

  std::cout << "nonlinearity entry is " << result << std::endl;

  // to destroy the pointer from dof_handler to fe:
  dof_handler.clear();
}
