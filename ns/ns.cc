#include "ns.h"

namespace POD
{
  namespace NavierStokes
  {

    NavierStokesRHS::NavierStokesRHS(FullMatrix<double> linear_operator,
                                     FullMatrix<double> mass_matrix,
                                     std::vector<FullMatrix<double>> nonlinear_operator,
                                     Vector<double> mean_contribution) :
      NonlinearOperatorBase(),
      linear_operator {linear_operator},
      nonlinear_operator {nonlinear_operator},
      mean_contribution {mean_contribution}
    {
      factorized_mass_matrix.reinit(mass_matrix.m());
      factorized_mass_matrix = mass_matrix;
      factorized_mass_matrix.compute_lu_factorization();
    }

    void NavierStokesRHS::apply(Vector<double> &dst, const Vector<double> &src)
    {
      const unsigned int n_dofs = src.size();
      linear_operator.vmult(dst, src);
      dst += mean_contribution;

      Vector<double> temp(n_dofs);
      // this could probably be parallelized with something like
      // #pragma omp parallel
      // {
      // Vector<double> temp;
      // #pragma omp parallel for
      // ...
      // }
      // }
      for (unsigned int pod_vector_n = 0; pod_vector_n < n_dofs; ++pod_vector_n)
        {
          nonlinear_operator[pod_vector_n].vmult(temp, src);
          dst(pod_vector_n) -= temp * src;
        }

      factorized_mass_matrix.apply_lu_factorization(dst, false);
    }
  }
}
