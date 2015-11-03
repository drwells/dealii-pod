#include <deal.II/lac/full_matrix.h>

#include <deal.II-pod/extra/extra.h>
#include <deal.II-pod/h5/h5.h>

int main()
{
  using namespace dealii;
  using namespace POD;

  extra::TemporaryFileName temporary_file_name;
  FullMatrix<double> full_matrix(10, 10);

  for (unsigned int i = 0; i < full_matrix.m(); ++i)
    {
      for (unsigned int j = 0; j < full_matrix.n(); ++j)
        {
          full_matrix(i, j) = double(i + j);
        }
    }
  H5::save_full_matrix(temporary_file_name.name, full_matrix);

  FullMatrix<double> other_full_matrix;
  H5::load_full_matrix(temporary_file_name.name, other_full_matrix);

  if(extra::are_equal(full_matrix, other_full_matrix, 1e-14))
    {
      return 0;
    }
  return 1;
}
