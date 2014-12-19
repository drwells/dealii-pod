#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_base.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "ns.h"

namespace POD
{
  namespace NavierStokes
  {
    template<int dim>
    class Chebyshev
    {
    public:
      Chebyshev(const dealii::Tensor<1, dim, int> od) : orders(od) {}

      double operator()(const dealii::Point<dim> point) const
      {
        double result = 1.0;
        for (unsigned int i = 0; i < dim; ++i)
          {
            result *= std::cos(orders[i]*std::acos(point[i]));
          }
        return result;
      }

    private:
      dealii::Tensor<1, dim, int> orders;
    };


    template<int dim>
    void get_chebyshev_vector(const dealii::Tensor<1, dim, int> orders,
                              const dealii::DoFHandler<dim> &dof_handler,
                              const dealii::FEValues<dim>   &fe_values,
                              dealii::Vector<double>        &coefficients)
    {
      coefficients.reinit(dof_handler.n_dofs());
      auto& mapping = fe_values.get_mapping();

      Chebyshev<dim> basis_function(orders);
      std::vector<dealii::Point<dim>> support_points(dof_handler.n_dofs());
      dealii::DoFTools::map_dofs_to_support_points(mapping,
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

constexpr unsigned int problem_dim = 2;

using namespace POD::NavierStokes;
int main(int argc, char** argv)
  {
    dealii::Triangulation<problem_dim> triangulation;
    dealii::GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (6);
    dealii::DoFHandler<problem_dim> dof_handler (triangulation);
    dealii::Tensor<1, problem_dim, int> orders;
    dealii::FE_Q<problem_dim> fe (3); // TODO this crashes > 1
    dealii::QGauss<problem_dim> quadrature (4);
    dealii::FEValues<problem_dim> fe_values(fe, quadrature,
                                    dealii::update_values
                                    | dealii::update_gradients
                                    | dealii::update_JxW_values);

    dof_handler.distribute_dofs(fe);
    std::cout << "DoFs: " << dof_handler.n_dofs() << std::endl;

    unsigned int n_vectors = 3;
    std::vector<dealii::Vector<double>> chebyshev_vectors;
    for (unsigned int i = 0; i < n_vectors; ++i)
      {
        for (unsigned int j = 0; j < problem_dim; ++j)
          {
            orders[j] = 3*j + 1;
          }
        dealii::Vector<double> chebyshev_vector;
        get_chebyshev_vector(orders, dof_handler, fe_values, chebyshev_vector);
        chebyshev_vectors.push_back(std::move(chebyshev_vector));
      }

    double result = compute_trilinearity(dof_handler, fe, fe_values, quadrature,
                                         chebyshev_vectors[0],
                                         chebyshev_vectors[1],
                                         chebyshev_vectors[2], 0);

    std::cout << "nonlinearity entry is " << result << std::endl;

    // to destroy the pointer from dof_handler to fe:
    dof_handler.clear();
  }
