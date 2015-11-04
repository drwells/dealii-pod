#include <deal.II/lac/full_matrix.h>

#include <deal.II-pod/extra/resize.h>

int main()
{
  using namespace dealii;
  using namespace POD;

  constexpr unsigned int original_length {10};
  FullMatrix<double> matrix(original_length, original_length);

  for (unsigned int i = 0; i < original_length; ++i)
    {
      for (unsigned int j = 0; j < original_length; ++j)
        {
          matrix(i, j) = double(i + j);
        }
    }

  constexpr unsigned int new_length {5};
  extra::resize(matrix, new_length);

  if (matrix.m() != new_length || matrix.n() != new_length)
    {
      return 1;
    }

  for (unsigned int i = 0; i < new_length; ++i)
    {
      for (unsigned int j = 0; j < new_length; ++j)
        {
          if (matrix(i, j) != double(i + j))
            {
              return 1;
            }
        }
    }

  return 0;
}
