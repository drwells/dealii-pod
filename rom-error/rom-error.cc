#include <deal.II/lac/block_vector.h>

#include <iostream>
#include <string>

#include "parameters.h"

#include "../pod/pod.h"
#include "../extra/extra.h"
#include "../h5/h5.h"

using namespace dealii;
{

Parameters::Parameters()
  {
    snapshot_file_names = "snapshot-*h5";
  }

int main()
{
  Parameters parameters;
  auto snapshot_file_names = extra::expand_file_names(parameters.snapshot_file_names);
  std::cout << "number of snapshots: " << snapshot_file_names.size() << std::endl;

  BlockVector<double> block_vector;
  for (auto &snapshot_file_name : snapshot_file_names)
    {
      H5::load_block_vector(snapshot_file_name, block_vector);
    }
}
