#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "../ns.h"

using namespace dealii;
using namespace POD::NavierStokes;

void test_l2_projection()
{
  const unsigned int n_pod_vectors = 10;
  const unsigned int cutoff_n = 5;
  FullMatrix<double> linear_operator(n_pod_vectors, n_pod_vectors);
  FullMatrix<double> mass_matrix(n_pod_vectors, n_pod_vectors);
  FullMatrix<double> joint_convection(n_pod_vectors, n_pod_vectors);
  std::vector<FullMatrix<double>> nonlinear_operator;
  {
    FullMatrix<double> identity(n_pod_vectors, n_pod_vectors);
    FullMatrix<double> values(n_pod_vectors, n_pod_vectors);
    FullMatrix<double> ones(n_pod_vectors, n_pod_vectors);
    for (unsigned int i = 0; i < n_pod_vectors; ++i)
      {
        for (unsigned int j = 0; j < n_pod_vectors; ++j)
          {
            values(i, j) = i + j;
            ones(i, j) = 1.0;
          }
      }

    for (unsigned int i = 0; i < n_pod_vectors; ++i)
      {
        identity(i, i) = 1.0;
      }

    mass_matrix = identity;
    for (unsigned int i = 0; i < n_pod_vectors; ++i)
      {
        nonlinear_operator.push_back(values);
        nonlinear_operator[i].add(static_cast<double>(i), ones);
      }
  }

  Vector<double> mean_contribution(n_pod_vectors);

  L2ProjectionFilterRHS projection_filter
    (linear_operator, mass_matrix, joint_convection,
     nonlinear_operator, mean_contribution, cutoff_n);

  Vector<double> src(n_pod_vectors), dst(n_pod_vectors);
  src = 1.0;
  projection_filter.apply(dst, src);

  for (auto &value : dst)
    {
      std::cout << value << std::endl;
    }
}

int main()
{
  test_l2_projection();
  return 0;
}
