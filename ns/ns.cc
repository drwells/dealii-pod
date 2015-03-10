#include "ns.h"

namespace POD
{
  namespace NavierStokes
  {
    PlainRHS::PlainRHS() {}


    PlainRHS::PlainRHS(const FullMatrix<double> linear_operator,
                       const FullMatrix<double> mass_matrix,
                       const std::vector<FullMatrix<double>> nonlinear_operator,
                       const Vector<double> mean_contribution) :
      linear_operator {linear_operator},
      nonlinear_operator {nonlinear_operator},
      mean_contribution {mean_contribution}
    {
      factorized_mass_matrix.reinit(mass_matrix.m());
      factorized_mass_matrix = mass_matrix;
      factorized_mass_matrix.compute_lu_factorization();
    }

    void PlainRHS::apply(Vector<double> &dst, const Vector<double> &src)
    {
      const unsigned int n_dofs = src.size();
      linear_operator.vmult(dst, src);
      dst += mean_contribution;

      Vector<double> temp(n_dofs);
      for (unsigned int pod_vector_n = 0; pod_vector_n < n_dofs; ++pod_vector_n)
        {
          nonlinear_operator[pod_vector_n].vmult(temp, src);
          dst(pod_vector_n) -= temp * src;
        }

      factorized_mass_matrix.apply_lu_factorization(dst, false);
    }


    PODDifferentialFilterRHS::PODDifferentialFilterRHS
    (const FullMatrix<double> linear_operator,
     const FullMatrix<double> mass_matrix,
     const FullMatrix<double> boundary_matrix,
     const FullMatrix<double> laplace_matrix,
     const std::vector<FullMatrix<double>> nonlinear_operator,
     const Vector<double> mean_contribution,
     const double filter_radius) :
      PlainRHS(linear_operator, mass_matrix, nonlinear_operator,
               mean_contribution),
      mass_matrix {mass_matrix}
    {
      FullMatrix<double> filter_matrix(mass_matrix.m());
      filter_matrix.add(filter_radius*filter_radius, laplace_matrix);
      filter_matrix.add(-1.0*filter_radius*filter_radius, boundary_matrix);
      filter_matrix.add(1.0, mass_matrix);
      factorized_filter_matrix.reinit(mass_matrix.m());
      factorized_filter_matrix.copy_from(filter_matrix);
      factorized_filter_matrix.compute_lu_factorization();
    }


    void PODDifferentialFilterRHS::apply
    (Vector<double> &dst, const Vector<double> &src)
    {
      const unsigned int n_dofs = src.size();
      linear_operator.vmult(dst, src);
      dst += mean_contribution;

      Vector<double> filtered_src(src.size());
      mass_matrix.vmult(filtered_src, src);
      factorized_filter_matrix.apply_lu_factorization(filtered_src, false);

      Vector<double> temp(n_dofs);
      for (unsigned int pod_vector_n = 0; pod_vector_n < n_dofs; ++pod_vector_n)
        {
          nonlinear_operator[pod_vector_n].vmult(temp, src);
          dst(pod_vector_n) -= temp * filtered_src;
        }

      factorized_mass_matrix.apply_lu_factorization(dst, false);
    }


    L2ProjectionFilterRHS::L2ProjectionFilterRHS
    (const FullMatrix<double> linear_operator,
     const FullMatrix<double> mass_matrix,
     const FullMatrix<double> joint_convection,
     const std::vector<FullMatrix<double>> nonlinear_operator,
     const Vector<double> mean_contribution,
     const unsigned int cutoff_n) :
      PlainRHS(linear_operator, mass_matrix, nonlinear_operator,
               mean_contribution),
      joint_convection {joint_convection},
      linear_operator_without_convection {linear_operator},
      cutoff_n {cutoff_n}
    {
      this->linear_operator_without_convection.add(-1.0, joint_convection);
    }


    void L2ProjectionFilterRHS::apply
    (Vector<double> &dst, const Vector<double> &src)
    {
      const unsigned int n_dofs = src.size();
      linear_operator_without_convection.vmult(dst, src);
      dst += mean_contribution;

      auto filtered_src = src;
      for (unsigned int pod_vector_n = cutoff_n; pod_vector_n < n_dofs;
           ++pod_vector_n)
        {
          filtered_src[pod_vector_n] = 0.0;
        }
      joint_convection.vmult_add(dst, filtered_src);

      Vector<double> temp(n_dofs);
      for (unsigned int pod_vector_n = 0; pod_vector_n < n_dofs; ++pod_vector_n)
        {
          nonlinear_operator[pod_vector_n].vmult(temp, src);
          dst(pod_vector_n) -= temp * filtered_src;
        }

      factorized_mass_matrix.apply_lu_factorization(dst, false);
    }


    PostDifferentialFilter::PostDifferentialFilter
    (const FullMatrix<double> &mass_matrix,
     const FullMatrix<double> &laplace_matrix,
     const FullMatrix<double> &boundary_matrix,
     const double filter_radius) :
      mass_matrix {mass_matrix}
    {
      FullMatrix<double> filter_matrix(mass_matrix.m());
      filter_matrix.add(1.0, mass_matrix);
      filter_matrix.add(filter_radius*filter_radius, laplace_matrix);
      filter_matrix.add(-1.0*filter_radius*filter_radius, boundary_matrix);
      factorized_post_filter_matrix.reinit(mass_matrix.m());
      factorized_post_filter_matrix.copy_from(filter_matrix);
      factorized_post_filter_matrix.compute_lu_factorization();
    }


    void PostDifferentialFilter::apply(Vector<double> &dst,
                                       const Vector<double> &src)
    {
      mass_matrix.vmult(dst, src);
      factorized_post_filter_matrix.apply_lu_factorization(dst, false);
    }


    PostL2ProjectionFilter::PostL2ProjectionFilter
    (const unsigned int cutoff_n) : cutoff_n (cutoff_n) {}


    void PostL2ProjectionFilter::apply(Vector<double> &dst,
                                       const Vector<double> &src)
    {
      dst = src;
      for (unsigned int pod_vector_n = cutoff_n; pod_vector_n < dst.size();
           ++pod_vector_n)
        {
          dst[pod_vector_n] = 0.0;
        }
    }
  }
}
