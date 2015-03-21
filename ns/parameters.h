#ifndef __deal2_rom_ns_parameters_h
#define __deal2_rom_ns_parameters_h
#include <deal.II/base/parameter_handler.h>

#include <fstream>

using namespace dealii;
namespace POD
{
  enum class FilterModel
  {Differential, L2Projection, PostDifferentialFilter, PostL2ProjectionFilter,
      LerayHybrid};

  namespace NavierStokes
  {
    class Parameters
    {
    public:
      Parameters();
      int outflow_label;
      double reynolds_n;
      bool renumber;
      int fe_order;

      POD::FilterModel filter_model;
      double filter_radius;
      unsigned int cutoff_n;
      bool filter_mean;

      double initial_time;
      double final_time;
      double time_step;

      int patch_refinement;
      double output_time_start;
      double output_time_stop;
      int output_interval;
      bool save_plot_pictures;

      void read_data(std::string file_name);
    protected:
      ParameterHandler parameter_handler;
    };
  }
}
#endif
