#ifndef __deal2_rom_compute_pod_parameters_h
#define __deal2_rom_compute_pod_parameters_h
#include <deal.II/base/parameter_handler.h>

#include <fstream>
#include <string>

using namespace dealii;
class Parameters
{
public:
  int fe_order;
  bool renumber;
  std::string triangulation_file_name;
  int outflow_label;

  std::string pod_vector_glob;
  std::string mean_vector_file_name;
  double filter_radius;

  void read_data(const std::string &file_name);
private:
  void configure_parameter_handler(ParameterHandler &file_name) const;
};
#endif
