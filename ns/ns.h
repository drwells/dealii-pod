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
    class PlainRHS : public ODE::OperatorBase
    {
    public:
      PlainRHS();
      PlainRHS(const FullMatrix<double> linear_operator,
               const FullMatrix<double> mass_matrix,
               const std::vector<FullMatrix<double>> nonlinear_operator,
               const Vector<double> mean_contribution);
      void apply(Vector<double> &dst, const Vector<double> &src) override;
    protected:
      const FullMatrix<double> linear_operator;
      LAPACKFullMatrix<double> factorized_mass_matrix;
      const std::vector<FullMatrix<double>> nonlinear_operator;
      const Vector<double> mean_contribution;
    };


    class PODDifferentialFilterRHS : public PlainRHS
    {
    public:
      PODDifferentialFilterRHS
      (const FullMatrix<double> linear_operator,
       const FullMatrix<double> mass_matrix,
       const FullMatrix<double> boundary_matrix,
       const FullMatrix<double> laplace_matrix,
       const std::vector<FullMatrix<double>> nonlinear_operator,
       const Vector<double> mean_contribution,
       const double filter_radius);
      void apply(Vector<double> &dst, const Vector<double> &src) override;
    private:
      const FullMatrix<double> mass_matrix;
      LAPACKFullMatrix<double> factorized_filter_matrix;
    };


    class L2ProjectionFilterRHS : public PlainRHS
    {
    public:
      L2ProjectionFilterRHS
      (const FullMatrix<double> linear_operator,
       const FullMatrix<double> mass_matrix,
       const FullMatrix<double> joint_convection,
       const std::vector<FullMatrix<double>> nonlinear_operator,
       const Vector<double> mean_contribution,
       const unsigned int cutoff_n);
      void apply(Vector<double> &dst, const Vector<double> &src) override;
    private:
      const FullMatrix<double> joint_convection;
      FullMatrix<double> linear_operator_without_convection;
      const unsigned int cutoff_n;
    };


    class PostDifferentialFilter : public ODE::OperatorBase
    {
    public:
      PostDifferentialFilter
      (const FullMatrix<double> &mass_matrix,
       const FullMatrix<double> &laplace_matrix,
       const FullMatrix<double> &boundary_matrix,
       const double filter_radius);
      virtual void apply(Vector<double> &dst, const Vector<double> &src);
    private:
      const FullMatrix<double> mass_matrix;
      LAPACKFullMatrix<double> factorized_post_filter_matrix;
    };


    class PostL2ProjectionFilter : public ODE::OperatorBase
    {
    public:
      PostL2ProjectionFilter(const unsigned int cutoff_n);
      virtual void apply(Vector<double> &dst, const Vector<double> &src);
    private:
      const unsigned int cutoff_n;
    };


    // The `static_cast` is dumb, but necessary to make the compiler
    // happy. There is surely a better way to do this with block sparse
    // matrices.
    template<int dim>
    using ArrayArray
    = std::array<std::array<SparseMatrix<double>, static_cast<size_t>(dim)>,
    static_cast<size_t>(dim)>;

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
     ArrayArray<dim> &gradient)
    {
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

          for (unsigned int row_n = 0; row_n < gradient.size(); ++row_n)
            {
              for (unsigned int derivative_n = 0; derivative_n < gradient[0].size();
                   ++derivative_n)
                {
                  // evaluate the derivative of the row component of the solution.
                  cell_matrix = 0.0;
                  local_gradient_values = 0.0;
                  for (unsigned int q = 0; q < quad.size(); ++q)
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          local_gradient_values[q] +=
                            solution.block(row_n)[local_indices[i]]
                            *fe_values.shape_grad(i, q)[derivative_n];
                        }
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              cell_matrix(i, j) += fe_values.shape_value(i, q)
                                                   *fe_values.shape_value(j, q)
                                                   *local_gradient_values[q]
                                                   *fe_values.JxW(q);
                            }
                        }
                    }

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          gradient[row_n][derivative_n].add
                          (local_indices[i], local_indices[j], cell_matrix(i, j));
                        }
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
    void create_reduced_nonlinearity
    (const DoFHandler<dim>                  &dof_handler,
     const SparsityPattern                  &sparsity_pattern,
     const QGauss<dim>                      &quad,
     const std::vector<BlockVector<double>> &pod_vectors,
     std::vector<FullMatrix<double>>        &nonlinear_operator)
    {
      create_reduced_nonlinearity
      (dof_handler, sparsity_pattern, quad, pod_vectors, pod_vectors,
       nonlinear_operator);
    }

    template<int dim>
    void create_reduced_nonlinearity
    (const DoFHandler<dim>                  &dof_handler,
     const SparsityPattern                  &sparsity_pattern,
     const QGauss<dim>                      &quad,
     const std::vector<BlockVector<double>> &pod_vectors,
     const std::vector<BlockVector<double>> &filtered_pod_vectors,
     std::vector<FullMatrix<double>>        &nonlinear_operator)
    {
      const unsigned int n_pod_dofs = pod_vectors.size();
      const unsigned int n_dofs = pod_vectors.at(0).block(0).size();
      nonlinear_operator.resize(0);
      for (unsigned int i = 0; i < n_pod_dofs; ++i)
        {
          nonlinear_operator.emplace_back(n_pod_dofs);
        }

      #pragma omp parallel for
      for (unsigned int j = 0; j < n_pod_dofs; ++j)
        {
          BlockVector<double> temp(dim, n_dofs);
          SparseMatrix<double> full_advection(sparsity_pattern);
          create_advective_linearization
          (dof_handler, quad, filtered_pod_vectors.at(j), full_advection);
          for (unsigned int k = 0; k < n_pod_dofs; ++k)
            {
              for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
                {
                  full_advection.vmult(temp.block(dim_n),
                                       pod_vectors.at(k).block(dim_n));
                }
              for (unsigned int i = 0; i < n_pod_dofs; ++i)
                {
                  nonlinear_operator[i](j, k) = pod_vectors.at(i)*temp;
                }
            }
        }
    }


    template<int dim>
    void create_nonlinear_centered_contribution
    (const DoFHandler<dim>            &dof_handler,
     const SparsityPattern            &sparsity_pattern,
     const QGauss<dim>                &quad,
     BlockVector<double>              &filtered_solution,
     BlockVector<double>              &solution,
     std::vector<BlockVector<double>> &pod_vectors,
     Vector<double>                   &contribution)
    {
      SparseMatrix<double> full_advection(sparsity_pattern);
      create_advective_linearization
      (dof_handler, quad, filtered_solution, full_advection);
      contribution.reinit(pod_vectors.size());

      BlockVector<double> right_vector(dim, pod_vectors.at(0).block(0).size());
      for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
        {
          full_advection.vmult(right_vector.block(dim_n), solution.block(dim_n));
        }
      for (unsigned int row = 0; row < pod_vectors.size(); ++row)
        {
          for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
            {
              contribution[row] +=
                pod_vectors.at(row).block(dim_n)*right_vector.block(dim_n);
            }
        }
    }


    template<int dim>
    void create_reduced_advective_linearization
    (const DoFHandler<dim>                  &dof_handler,
     const SparsityPattern                  &sparsity_pattern,
     const QGauss<dim>                      &quad,
     const BlockVector<double>              &solution,
     const std::vector<BlockVector<double>> &pod_vectors,
     FullMatrix<double>                     &advection)
    {
      SparseMatrix<double> full_advection(sparsity_pattern);
      create_advective_linearization(dof_handler, quad, solution, full_advection);
      advection.reinit(pod_vectors.size(), pod_vectors.size());

      BlockVector<double> temp(dim, pod_vectors.at(0).block(0).size());
      for (unsigned int j = 0; j < pod_vectors.size(); ++j)
        {
          for (unsigned int dim_n = 0; dim_n < dim; ++dim_n)
            {
              full_advection.vmult(temp.block(dim_n),
                                   pod_vectors.at(j).block(dim_n));
            }
          #pragma omp parallel for
          for (unsigned int i = 0; i < pod_vectors.size(); ++i)
            {
              advection(i, j) = pod_vectors.at(i) * temp;
            }
        }
    }


    template<int dim>
    void create_reduced_gradient_linearization
    (const DoFHandler<dim>                  &dof_handler,
     const SparsityPattern                  &sparsity_pattern,
     const QGauss<dim>                      &quad,
     const BlockVector<double>              &solution,
     const std::vector<BlockVector<double>> &pod_vectors,
     FullMatrix<double>                     &gradient)
    {
      create_reduced_gradient_linearization
      (dof_handler, sparsity_pattern, quad, solution, pod_vectors,
       pod_vectors, gradient);
    }


    template<int dim>
    void create_reduced_gradient_linearization
    (const DoFHandler<dim>                  &dof_handler,
     const SparsityPattern                  &sparsity_pattern,
     const QGauss<dim>                      &quad,
     const BlockVector<double>              &solution,
     const std::vector<BlockVector<double>> &pod_vectors,
     const std::vector<BlockVector<double>> &filtered_pod_vectors,
     FullMatrix<double>                     &gradient)
    {
      ArrayArray<dim> gradient_matrices;
      for (auto &row : gradient_matrices)
        {
          for (auto &matrix : row)
            {
              matrix.reinit(sparsity_pattern);
            }
        }
      create_gradient_linearization(dof_handler, quad, solution,
                                    gradient_matrices);
      gradient.reinit(pod_vectors.size(), pod_vectors.size());

      BlockVector<double> temp(dim, pod_vectors.at(0).block(0).size());
      for (unsigned int j = 0; j < pod_vectors.size(); ++j)
        {
          temp = 0.0;
          auto &rhs_vector = filtered_pod_vectors.at(j);
          for (unsigned int row_n = 0; row_n < gradient_matrices.size();
               ++row_n)
            {
              for (unsigned int column_n = 0;
                   column_n < gradient_matrices[0].size();
                   ++column_n)
                {
                  gradient_matrices[row_n][column_n].vmult_add
                  (temp.block(row_n), rhs_vector.block(column_n));
                }
            }
          for (unsigned int i = 0; i < pod_vectors.size(); ++i)
            {
              auto &lhs_vector = pod_vectors.at(i);
              gradient(i, j) = lhs_vector * temp;
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
