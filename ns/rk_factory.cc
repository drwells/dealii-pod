#include "rk_factory.h"

namespace POD
{
  namespace NavierStokes
  {
    using namespace dealii;

    std::pair<std::string, std::unique_ptr<ODE::RungeKuttaBase>> rk_factory
    (const FullMatrix<double>              &boundary_matrix,
     const FullMatrix<double>              &joint_convection,
     const FullMatrix<double>              &laplace_matrix,
     const FullMatrix<double>              &linear_operator,
     const Vector<double>                  &mean_contribution_vector,
     const FullMatrix<double>              &mass_matrix,
     const std::vector<FullMatrix<double>> &nonlinear_operator,
     const POD::NavierStokes::Parameters   &parameters)
    {
      std::ostringstream outname;
      outname.precision(8);
      std::ostringstream outname_tail;
      if (not parameters.filter_mean)
        {
          outname_tail << "-unfiltered-mean";
        }
      outname_tail << "-r-" << mean_contribution_vector.size() // n_pod_dofs
                   << "-Re-" << parameters.reynolds_n
                   << ".h5";
      std::unique_ptr<POD::NavierStokes::PlainRHS> plain_rhs_function
        (new POD::NavierStokes::PlainRHS
         (linear_operator, mass_matrix, nonlinear_operator,
          mean_contribution_vector));
      std::unique_ptr<ODE::RungeKutta4> rk_method {new ODE::RungeKutta4()};
      if (parameters.filter_model == POD::FilterModel::Differential)
        {
          outname << "pod-leray-radius-" << parameters.filter_radius;
          rk_method = std::unique_ptr<ODE::RungeKutta4>
            (new ODE::RungeKutta4(std::move(plain_rhs_function)));
        }
      else if (parameters.filter_model == POD::FilterModel::L2Projection
               or parameters.filter_model == POD::FilterModel::LerayHybrid)
        {
          if (parameters.filter_mean
              and parameters.filter_model == POD::FilterModel::L2Projection)
            {
              Assert(false, StandardExceptions::ExcNotImplemented());
            }
          if (parameters.filter_model == POD::FilterModel::L2Projection)
            {
              outname << "pod-l2-projection-cutoff-" << parameters.cutoff_n;
            }
          else
            {
              outname << "pod-leray-hybrid-cutoff-" << parameters.cutoff_n
                      << "-radius-" << parameters.filter_radius;
            }
          std::unique_ptr<POD::NavierStokes::L2ProjectionFilterRHS> rhs_function
            (new POD::NavierStokes::L2ProjectionFilterRHS
             (linear_operator, mass_matrix, joint_convection, nonlinear_operator,
              mean_contribution_vector, parameters.cutoff_n));
          rk_method = std::unique_ptr<ODE::RungeKutta4>
            (new ODE::RungeKutta4(std::move(rhs_function)));
        }
      else if (parameters.filter_model == POD::FilterModel::PostDifferentialFilter)
        {
          if (parameters.filter_mean)
            {
              Assert(false, StandardExceptions::ExcNotImplemented());
            }
          outname << "pod-postfilter-differential-radius-"
                  << parameters.filter_radius;
          std::unique_ptr<POD::NavierStokes::PostDifferentialFilter> filter_function
            (new POD::NavierStokes::PostDifferentialFilter
             (mass_matrix, laplace_matrix, boundary_matrix,
              parameters.filter_radius));
          rk_method = std::unique_ptr<ODE::RungeKutta4PostFilter>
            (new ODE::RungeKutta4PostFilter
             (std::move(plain_rhs_function), std::move(filter_function)));
        }
      else if (parameters.filter_model == POD::FilterModel::PostL2ProjectionFilter)
        {
          if (parameters.filter_mean)
            {
              StandardExceptions::ExcNotImplemented();
            }
          outname << "pod-postfilter-cutoff-n-" << parameters.cutoff_n;
          std::unique_ptr<POD::NavierStokes::PostL2ProjectionFilter> filter_function
            (new POD::NavierStokes::PostL2ProjectionFilter
             (parameters.cutoff_n));
          rk_method = std::unique_ptr<ODE::RungeKutta4PostFilter>
            (new ODE::RungeKutta4PostFilter
             (std::move(plain_rhs_function), std::move(filter_function)));
        }
      else if (parameters.filter_model == POD::FilterModel::ADLavrentiev)
        {
          if (!parameters.filter_mean)
            {
              StandardExceptions::ExcNotImplemented();
            }
          outname << "pod-ad-lavrentiev-" << parameters.lavrentiev_parameter
                  << "-noise-multiplier-" << parameters.noise_multiplier;

          std::unique_ptr<POD::NavierStokes::AD::FilterBase> ad_filter
            (new POD::NavierStokes::AD::LavrentievFilter
             (mass_matrix, laplace_matrix, boundary_matrix,
              parameters.filter_radius, parameters.noise_multiplier,
              parameters.lavrentiev_parameter));
          std::unique_ptr<POD::NavierStokes::AD::FilterRHS> rhs_function
            (new POD::NavierStokes::AD::FilterRHS
             (mass_matrix, boundary_matrix, laplace_matrix, joint_convection,
              nonlinear_operator, mean_contribution_vector,
              parameters.reynolds_n, std::move(ad_filter)));
          rk_method = std::unique_ptr<ODE::RungeKutta4>
            (new ODE::RungeKutta4(std::move(rhs_function)));
        }
      else
        {
          StandardExceptions::ExcNotImplemented();
        }
      outname << outname_tail.str();

      return std::make_pair(outname.str(), std::move(rk_method));
    }
  }
}
