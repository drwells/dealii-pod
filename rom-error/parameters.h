#ifndef __deal2_rom_rom_error_h
#define __deal2_rom_rom_error_h

#include <string>

class Parameters
{
public:
  std::string snapshot_glob;
  std::string pod_vector_glob;
  std::string mean_vector_file_name;
  std::string pod_coefficients_file_name;
  bool renumber;
  unsigned int fe_order;

  double snapshot_start_time;
  double snapshot_stop_time;

  double rom_start_time;
  double rom_stop_time;
  Parameters();
};

#endif
