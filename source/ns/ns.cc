#include <deal.II-pod/ns/ns.h>

namespace POD
{
  using namespace dealii;

  namespace NavierStokes
  {
    PlainRHS::PlainRHS() :
      n_pod_dofs {numbers::invalid_unsigned_int}
    {}


    PlainRHS::PlainRHS(const FullMatrix<double> linear_operator,
                       const FullMatrix<double> mass_matrix,
                       const std::vector<FullMatrix<double>> nonlinear_operator,
                       const Vector<double> mean_contribution) :
      linear_operator {linear_operator},
      nonlinear_operator {nonlinear_operator},
      mean_contribution {mean_contribution},
      n_pod_dofs {mass_matrix.m()},
      temp(n_pod_dofs)
    {
      factorized_mass_matrix.reinit(mass_matrix.m());
      factorized_mass_matrix = mass_matrix;
      factorized_mass_matrix.compute_lu_factorization();
    }

    void PlainRHS::apply(Vector<double> &dst, const Vector<double> &src)
    {
      linear_operator.vmult(dst, src);
      dst += mean_contribution;

      for (unsigned int pod_vector_n = 0; pod_vector_n < n_pod_dofs; ++pod_vector_n)
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


    namespace AD
    {
      FilterBase::FilterBase
      (const FullMatrix<double> mass_matrix,
       const FullMatrix<double> laplace_matrix,
       const FullMatrix<double> boundary_matrix,
       const double filter_radius,
       const double noise_multiplier) :
        mass_matrix {mass_matrix},
        laplace_matrix {laplace_matrix},
        boundary_matrix {boundary_matrix},
        filter_radius {filter_radius},
        noise_multiplier {noise_multiplier},
        distribution(0.0, 1.0)
      {
        filter_matrix = mass_matrix;
        filter_matrix.add(filter_radius*filter_radius, laplace_matrix);
        filter_matrix.add(-1.0*filter_radius*filter_radius, boundary_matrix);

        factorized_mass_matrix.reinit(mass_matrix.m());
        factorized_mass_matrix.copy_from(mass_matrix);
        factorized_mass_matrix.compute_lu_factorization();

        factorized_filter_matrix.reinit(mass_matrix.m());
        factorized_filter_matrix.copy_from(filter_matrix);
        factorized_filter_matrix.compute_lu_factorization();
      }


      LavrentievFilter::LavrentievFilter
      (const FullMatrix<double> mass_matrix,
       const FullMatrix<double> laplace_matrix,
       const FullMatrix<double> boundary_matrix,
       const double filter_radius,
       const double noise_multiplier,
       const double lavrentiev_parameter) :
        FilterBase(mass_matrix, laplace_matrix, boundary_matrix, filter_radius,
                   noise_multiplier),
        lavrentiev_parameter {lavrentiev_parameter},
        work0(mass_matrix.m()),
        work1(mass_matrix.m())
      {}

      /*
       * Apply the filter G = (M + d^2 S)^-1 to the input vector src. src
       * should be in the physical (not filtered) space.
       */
      void LavrentievFilter::apply
      (Vector<double> &dst, const Vector<double> &src)
      {
        // TODO is there anything specific to Lavrentiev that should be done here?
        dst = src;
        factorized_filter_matrix.apply_lu_factorization(dst, false);
      }


      /*
       * Solve the linear system
       *
       * (M + d^2 S) u^{AD-L} = (M + mu M + mu d^2 S) M (\bar{u} + noise)
       *
       * for u^{AD-L}. This approximately undoes the action of the filter.
       */
      void LavrentievFilter::apply_inverse
      (Vector<double> &dst, const Vector<double> &src)
      {
        work0 = src;

        for (unsigned int i = 0; i < work0.size(); ++i)
          {
            work0[i] += noise_multiplier * distribution(generator);
          }

        // M (\bar{u} + noise) == work1
        mass_matrix.vmult(work1, work0);
        mass_matrix.vmult(dst, work1);

        filter_matrix.vmult(work0, work1);
        work0 *= lavrentiev_parameter;
        dst += work0;

        factorized_filter_matrix.apply_lu_factorization(dst, false);
      }

      FilterRHS::FilterRHS
      (const FullMatrix<double> mass_matrix,
       const FullMatrix<double> boundary_matrix,
       const FullMatrix<double> laplace_matrix,
       const FullMatrix<double> joint_convection_matrix,
       const std::vector<FullMatrix<double>> nonlinear_operator,
       const Vector<double> mean_contribution,
       const double reynolds_n,
       std::unique_ptr<FilterBase> ad_filter) :
        mass_matrix {mass_matrix},
        boundary_matrix {boundary_matrix},
        laplace_matrix {laplace_matrix},
        joint_convection_matrix {joint_convection_matrix},
        nonlinear_operator {nonlinear_operator},
        mean_contribution {mean_contribution},
        reynolds_n {reynolds_n},
        unfiltered_contribution(mass_matrix.m()),
        work1(mass_matrix.m()),
        work2(mass_matrix.m()),
        approximately_deconvolved_solution(mass_matrix.m()),
        filter {std::move(ad_filter)}
      {
        factorized_mass_matrix.reinit(mass_matrix.m());
        factorized_mass_matrix = mass_matrix;
        factorized_mass_matrix.compute_lu_factorization();
      }


      void FilterRHS::apply
      (Vector<double> &dst, const Vector<double> &src)
      {
        dst = 0.0;
        // get an approximation of the unfiltered solution
        filter->apply_inverse(approximately_deconvolved_solution, src);

        // compute the result of the filtered mean contribution
        unfiltered_contribution = mean_contribution;

        // add the result of the convection matrices
        joint_convection_matrix.vmult(work1, approximately_deconvolved_solution);
        unfiltered_contribution += work1;

        // and the nonlinearity
        for (unsigned int pod_vector_n = 0; pod_vector_n < src.size(); ++pod_vector_n)
          {
            nonlinear_operator[pod_vector_n].vmult(work1, approximately_deconvolved_solution);
            unfiltered_contribution[pod_vector_n] -= work1 * approximately_deconvolved_solution;
          }
        factorized_mass_matrix.apply_lu_factorization(unfiltered_contribution, false);
        filter->apply(work1, unfiltered_contribution);
        dst += work1;

        // add the result of the laplace matrix
        laplace_matrix.vmult(work1, src);
        dst.add(-1.0/reynolds_n, work1);

        // and the boundary matrix
        boundary_matrix.vmult(work1, src);
        dst.add(1.0/reynolds_n, work1);

        factorized_mass_matrix.apply_lu_factorization(dst, false);
      }
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


    PostDifferentialFilterRelax::PostDifferentialFilterRelax
    (const FullMatrix<double> &mass_matrix,
     const FullMatrix<double> &laplace_matrix,
     const FullMatrix<double> &boundary_matrix,
     const double filter_radius,
     const double relaxation_parameter) :
      relaxation_parameter(relaxation_parameter),
      differential_filter(mass_matrix, laplace_matrix, boundary_matrix, filter_radius)
    {}


    void PostDifferentialFilterRelax::apply(Vector<double> &dst,
                                            const Vector<double> &src)
    {
      differential_filter.apply(dst, src);
      dst.sadd((1.0 - relaxation_parameter), relaxation_parameter, src);
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
