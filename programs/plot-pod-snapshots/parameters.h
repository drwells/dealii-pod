#include <deal.II/base/parameter_handler.h>

#include <string>

namespace POD
{
  using namespace dealii;

  class Parameters
  {
  public:
    unsigned int dimension;
    unsigned int fe_order;
    std::string triangulation_file_name;
    bool renumber;

    std::string pod_coefficients_file_name;
    double start_time;
    double time_step;

    unsigned int output_interval;

    void read_data(const std::string &file_name);
  private:
    void configure_parameter_handler(ParameterHandler &file_name) const;
  };
}
