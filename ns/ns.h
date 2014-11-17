#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/vector.h>

namespace POD
{
  namespace NavierStokes
  {
    template<int dim>
    double compute_trilinearity(dealii::DoFHandler<dim> &dof_handler,
                                dealii::FE_Q<dim>       &fe,
                                dealii::FEValues<dim>   &fe_values,
                                dealii::QGauss<dim>     &quadrature_formula,
                                dealii::Vector<double>  &pod_vector_0,
                                dealii::Vector<double>  &pod_vector_1,
                                dealii::Vector<double>  &pod_vector_2,
                                int derivative_number)
      {
        Assert(derivative_number < dim, dealii::ExcInternalError());
        unsigned int dofs_per_cell = fe.dofs_per_cell;
        std::vector<dealii::types::global_dof_index> local_dof_indices
          (dofs_per_cell);
        std::vector<double>
                local_coeffs_0(dofs_per_cell),
                local_coeffs_1(dofs_per_cell),
                local_coeffs_2(dofs_per_cell);
        double result = 0;

        auto cell = dof_handler.begin_active();
        decltype(cell) endc = dof_handler.end();
        for (; cell != endc; ++cell)
          {
            fe_values.reinit (cell);
            cell->get_dof_indices (local_dof_indices);
            unsigned int i = 0;
            for (auto global_index : local_dof_indices)
              {
                local_coeffs_0[i] = pod_vector_0[global_index];
                local_coeffs_1[i] = pod_vector_1[global_index];
                local_coeffs_2[i] = pod_vector_2[global_index];
                ++i;
              }

            double cell_integral = 0;
            for (unsigned int q_index = 0; q_index < quadrature_formula.size();
                 ++q_index)
              {
                double value_i = 0.0;
                double value_j = 0.0;
                double value_k = 0.0;
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    value_i += local_coeffs_0[i]*fe_values.shape_value (i, q_index);
                    value_j += local_coeffs_1[i]*fe_values.shape_value (i, q_index);
                    value_k += local_coeffs_2[i]
                      *fe_values.shape_grad(i, q_index)[derivative_number];
                  }

                cell_integral += fe_values.JxW(q_index)*value_i*value_j*value_k;
              }
            result += cell_integral;
          }

        return result;
      }
  }
}
