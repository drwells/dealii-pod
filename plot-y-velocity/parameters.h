#ifndef __deal2_rom_plot_y_velocity_parameters_h
#define __deal2_rom_plot_y_velocity_parameters_h
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
  double time_step;

  int patch_level;

  void read_data(const std::string &file_name);
private:
  void configure_parameter_handler(ParameterHandler &file_name) const;
};
#endif
