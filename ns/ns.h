#ifndef __deal2_ns_pod_h
#define __deal2_ns_pod_h
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/fe/fe_values.h>

#include <array>
#include <iostream>
#include <limits>
#include <vector>

#include "../ode/ode.h"

using namespace dealii;
namespace POD
{
  constexpr double NaN = std::numeric_limits<double>::quiet_NaN();


  namespace NavierStokes
  {
    class NavierStokesRHS : public ODE::NonlinearOperatorBase
    {
    public:
      NavierStokesRHS(FullMatrix<double> linear_operator,
                      FullMatrix<double> mass_matrix,
                      std::vector<FullMatrix<double>> nonlinear_operator,
                      Vector<double> mean_contribution);
      void apply(Vector<double> &dst, const Vector<double> &src) override;
    private:
      FullMatrix<double> linear_operator;
      LAPACKFullMatrix<double> factorized_mass_matrix;
      std::vector<FullMatrix<double>> nonlinear_operator;
      Vector<double> mean_contribution;
    };

    class NavierStokesLerayRegularizationRHS : public NavierStokesRHS
    {
    public:
      NavierStokesLerayRegularizationRHS
      (FullMatrix<double> linear_operator,
       FullMatrix<double> mass_matrix,
       FullMatrix<double> laplace_matrix,
       std::vector<FullMatrix<double>> nonlinear_operator,
       Vector<double> mean_contribution,
       double filter_radius);
      void apply(Vector<double> &dst, const Vector<double> &src) override;
    private:
      FullMatrix<double> linear_operator;
      FullMatrix<double> laplace_matrix;
      LAPACKFullMatrix<double> factorized_mass_matrix;
      std::vector<FullMatrix<double>> nonlinear_operator;
      Vector<double> mean_contribution;
      double filter_radius;
    };

    template<int dim>
    double trilinearity_term(
      const QGauss<dim>         &quad,
      const DoFHandler<dim>     &dof_handler,
      const BlockVector<double> &pod_vector_0,
      const BlockVector<double> &pod_vector_1,
      const BlockVector<double> &pod_vector_2)
    {
      double result = 0.0;

      auto &fe = dof_handler.get_fe();
      FEValues<dim> fe_values(fe, quad, update_values | update_gradients |
                              update_JxW_values);
      unsigned int const dofs_per_cell = fe.dofs_per_cell;
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      std::vector<double> local_test_coeffs(dofs_per_cell, 0.0);
      std::vector<double> local_grad_coeffs(dofs_per_cell, 0.0);
      std::vector<std::vector<double>> local_convection_coeffs;
      for (unsigned int i = 0; i < dim; ++i)
        {
          local_convection_coeffs.emplace_back(dofs_per_cell, POD::NaN);
        }

      auto cell = dof_handler.begin_active(),
           endc = dof_handler.end();
      for (; cell != endc; ++cell)
        {
          for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
            {
              fe_values.reinit(cell);
              cell->get_dof_indices(local_dof_indices);
              unsigned int i = 0;
              for (auto global_index : local_dof_indices)
                {
                  local_test_coeffs[i] = pod_vector_0.block(dim_n)[global_index];
                  local_grad_coeffs[i] = pod_vector_2.block(dim_n)[global_index];
                  for (unsigned int j = 0; j < dim; ++j)
                    {
                      local_convection_coeffs[j][i] = pod_vector_1.block(j)[global_index];
                    }
                  ++i;
                }

              double cell_integral = 0.0;
              for (unsigned int q = 0; q < quad.size(); ++q)
                {
                  double point_value = 0.0;
                  double test_value = 0.0;
                  std::array<double, dim> convective_values {0.0};
                  std::array<double, dim> convective_grads {0.0};

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      test_value += fe_values.shape_value(i, q)*local_test_coeffs[i];
                      for (unsigned int j = 0; j < dim; ++j)
                        {
                          convective_values[j] +=
                            fe_values.shape_value(i, q)*local_convection_coeffs[j][i];
                          convective_grads[j] +=
                            fe_values.shape_grad(i, q)[j]*local_grad_coeffs[i];
                        }
                    }
                  for (unsigned int j = 0; j < dim; ++j)
                    {
                      point_value += convective_values[j]*convective_grads[j];
                    }
                  point_value *= test_value;
                  cell_integral += fe_values.JxW(q)*point_value;
                }
              result += cell_integral;
            }
        }
      return result;
    }


    template<int dim>
    void create_gradient_linearization
    (const DoFHandler<dim>     &dof_handler,
     const QGauss<dim>         &quad,
     const BlockVector<double> &solution,
     // The `static_cast` is dumb, but necessary to make the compiler happy.
     std::array<SparseMatrix<double>, static_cast<size_t>(dim)> &gradient)
    {
      ExcInternalError();
      auto &fe = dof_handler.get_fe();
      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      FEValues<dim> fe_values(fe, quad, update_values | update_gradients
                              | update_JxW_values);

      std::vector<types::global_dof_index> local_indices(dofs_per_cell);
      Vector<double> local_gradient_values(quad.size());

      typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

      for (; cell != endc; ++cell)
        {
          fe_values.reinit(cell);
          cell->get_dof_indices(local_indices);

          for (unsigned int derivative_n = 0; derivative_n < dim; ++derivative_n)
            {
              cell_matrix = 0.0;
              local_gradient_values = 0.0;
              for (unsigned int q = 0; q < quad.size(); ++q)
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
                        {
                          local_gradient_values[q] +=
                            solution.block(dim_n)[local_indices[i]]
                            *fe_values.shape_grad(i, q)[derivative_n];
                        }
                    }
                }

              for (unsigned int q = 0; q < quad.size(); ++q)
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          cell_matrix(i, j) +=
                            fe_values.shape_value(i, q)
                            *local_gradient_values[q]
                            *fe_values.shape_value(j, q)
                            *fe_values.JxW(q);
                        }
                    }
                }
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      gradient[derivative_n].add
                      (local_indices[i], local_indices[j], cell_matrix(i, j));
                    }
                }
            }
        }
    }


    template<int dim>
    void create_advective_linearization(const DoFHandler<dim>     &dof_handler,
                                        const QGauss<dim>         &quad,
                                        const BlockVector<double> &solution,
                                        SparseMatrix<double>      &advection)
    {
      // This function passes (simple) tests but gives wrong results in
      // practice. I do not yet know why.
      ExcInternalError();
      auto &fe = dof_handler.get_fe();
      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      FEValues<dim> fe_values(fe, quad, update_values | update_gradients
                              | update_JxW_values);

      std::vector<types::global_dof_index> local_indices(dofs_per_cell);
      std::array<Vector<double>, dim> local_advection_values;
      for (auto &vector : local_advection_values)
        {
          vector.reinit(quad.size());
        }

      typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

      for (; cell != endc; ++cell)
        {
          fe_values.reinit(cell);
          cell_matrix = 0.0;
          cell->get_dof_indices(local_indices);
          for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
            {
              local_advection_values[dim_n] = 0.0;
              for (unsigned int q = 0; q < quad.size(); ++q)
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      local_advection_values[dim_n][q] +=
                        fe_values.shape_value(i, q)
                        *solution.block(dim_n)[local_indices[i]];
                    }
                }
            }

          for (unsigned int q = 0; q < quad.size(); ++q)
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
                        {
                          cell_matrix(i, j) += fe_values.shape_value(i, q)
                                               *local_advection_values[dim_n][q]
                                               *fe_values.shape_grad(j, q)[dim_n]
                                               *fe_values.JxW(q);
                        }
                    }
                }
            }

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  advection.add(local_indices[i], local_indices[j], cell_matrix(i, j));
                }
            }
        }
    }


    template<int dim>
    void create_boundary_matrix(const DoFHandler<dim> &dof_handler,
                                const QGauss<dim - 1> &face_quad,
                                const unsigned int outflow_label,
                                SparseMatrix<double> &boundary_matrix)
    {
      auto &fe = dof_handler.get_fe();
      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      std::vector<types::global_dof_index> local_indices(dofs_per_cell);
      FEFaceValues<dim> fe_face_values(fe, face_quad, update_values |
                                       update_gradients | update_JxW_values);

      typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

      for (; cell != endc; ++cell)
        {
          cell_matrix = 0;
          cell->get_dof_indices(local_indices);

          for (unsigned int face_n = 0; face_n < GeometryInfo<dim>::faces_per_cell;
               ++face_n)
            {
              if (cell->face(face_n)->at_boundary()
                  && cell->face(face_n)->boundary_indicator() == outflow_label)
                {
                  fe_face_values.reinit(cell, face_n);
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      // Note that even if the jth basis function does not have
                      // support on a face then its derivative may have support.
                      if (fe.has_support_on_face(i, face_n))
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              for (unsigned int q = 0; q < face_quad.size(); ++q)
                                {
                                  cell_matrix(i, j) +=
                                    fe_face_values.shape_value(i, q) *
                                    fe_face_values.shape_grad(j, q)[0] *
                                    fe_face_values.JxW(q);
                                }
                            }
                        }
                    }
                }
            }
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  boundary_matrix.add(local_indices[i], local_indices[j],
                                      cell_matrix(i, j));
                }
            }
        }
    }
  }
}
#endif