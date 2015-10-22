#ifndef dealii__rom_compute_pod_parameters_h
#define dealii__rom_compute_pod_parameters_h
#include <deal.II/base/parameter_handler.h>

#include <fstream>
#include <string>

using namespace dealii;
class Parameters
{
public:
  int fe_order;
  bool renumber;
  std::string snapshot_glob;
  std::string triangulation_file_name;

  int n_pod_vectors;
  bool center_trajectory;

  bool save_mass_matrix;
  bool save_laplace_matrix;
  bool save_plot_pictures;

  void read_data(const std::string &file_name);
private:
  void configure_parameter_handler(ParameterHandler &file_name) const;
};
#endif
