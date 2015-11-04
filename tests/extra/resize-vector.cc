#include <deal.II/lac/vector.h>

#include <deal.II-pod/extra/resize.h>

int main()
{
  using namespace dealii;
  using namespace POD;

  constexpr unsigned int original_length {10};
  Vector<double> vector(original_length);

  for (unsigned int i = 0; i < original_length; ++i)
    {
      vector[i] = double(i);
    }

  constexpr unsigned int new_length {5};
  extra::resize(vector, new_length);

  if (vector.size() != new_length)
    {
      return 1;
    }

  for (unsigned int i = 0; i < new_length; ++i)
    {
      if (vector[i] != double(i))
        {
          return 1;
        }
    }

  return 0;
}
