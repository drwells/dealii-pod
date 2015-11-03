#include <deal.II/lac/block_vector.h>

#include <deal.II-pod/extra/extra.h>
#include <deal.II-pod/h5/h5.h>

int main()
{
  using namespace dealii;
  using namespace POD;

  extra::TemporaryFileName temporary_file_name;
  BlockVector<double> block_vector(3, 10);

  for (unsigned int i = 0; i < block_vector.n_blocks(); ++i)
    {
      for (unsigned int j = 0; j < block_vector.block(0).size(); ++j)
        {
          block_vector.block(i)[j] = double(i + j);
        }
    }
  H5::save_block_vector(temporary_file_name.name, block_vector);

  BlockVector<double> other_block_vector;
  H5::load_block_vector(temporary_file_name.name, other_block_vector);

  if(extra::are_equal(block_vector, other_block_vector, 1e-14))
    {
      return 0;
    }
  return 1;
}
