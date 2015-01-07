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
      for (unsigned int pod_vector_n = 0; pod_vector_n < n_dofs; ++pod_vector_n)
        {
          nonlinear_operator[pod_vector_n].vmult(temp, src);
          dst(pod_vector_n) -= temp * src;
        }

      factorized_mass_matrix.apply_lu_factorization(dst, false);
    }

    // template specializations
    template void create_advective_linearization<2>
    (const DoFHandler<2>       &dof_handler,
     const QGauss<2>           &quad,
     const BlockVector<double> &solution,
     SparseMatrix<double>      &advection);

    template void create_advective_linearization<3>
    (const DoFHandler<3>       &dof_handler,
     const QGauss<3>           &quad,
     const BlockVector<double> &solution,
     SparseMatrix<double>      &advection);

    template void create_gradient_linearization<2>
    (const DoFHandler<2>                 &dof_handler,
     const QGauss<2>                     &quad,
     const BlockVector<double>           &solution,
     std::array<SparseMatrix<double>, 2> &gradient);

    template void create_gradient_linearization<3>
    (const DoFHandler<3>                 &dof_handler,
     const QGauss<3>                     &quad,
     const BlockVector<double>           &solution,
     std::array<SparseMatrix<double>, 3> &gradient);

    template double trilinearity_term<2>(
      const QGauss<2>         &quad,
      const DoFHandler<2>     &dof_handler,
      const BlockVector<double> &pod_vector_0,
      const BlockVector<double> &pod_vector_1,
      const BlockVector<double> &pod_vector_2);

    template double trilinearity_term<3>(
      const QGauss<3>         &quad,
      const DoFHandler<3>     &dof_handler,
      const BlockVector<double> &pod_vector_0,
      const BlockVector<double> &pod_vector_1,
      const BlockVector<double> &pod_vector_2);

    template void create_boundary_matrix<2>
    (const DoFHandler<2> &dof_handler,
     const QGauss<1> &face_quad,
     const unsigned int outflow_label,
     SparseMatrix<double> &boundary_matrix);

    template void create_boundary_matrix<3>
    (const DoFHandler<3> &dof_handler,
     const QGauss<2> &face_quad,
     const unsigned int outflow_label,
     SparseMatrix<double> &boundary_matrix);
  }
}