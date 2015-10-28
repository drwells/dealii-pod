#include "parameters.h"

namespace POD
{
  namespace NavierStokes
  {
    void Parameters::configure_parameter_handler
    (ParameterHandler &parameter_handler) const
    {
      parameter_handler.enter_subsection("DNS Information");
      {
        parameter_handler.declare_entry
          ("reynolds_n", "50.0", Patterns::Double(0.0), " Reynolds number of the "
           "underlying simulation.");
      }
      parameter_handler.leave_subsection();

      parameter_handler.enter_subsection("Filtering Model Configuration");
      {
        parameter_handler.declare_entry
          ("filter_model", "Differential",
           Patterns::Selection
           ("L2Projection|Differential|PostL2ProjectionFilter|PostDifferentialFilter"
            "|LerayHybrid|ADLavrentiev|ADTikhonov"),
           " Filtering strategy for the POD-ROM.");
        parameter_handler.declare_entry
          ("noise_multiplier", "0.0", Patterns::Double(0.0), "Multiplier on the "
           "randomly generated noise vector for approximate deconvolution models.");
        parameter_handler.declare_entry
          ("lavrentiev_parameter", "0.0", Patterns::Double(0.0), "Multiplier for"
           " the Lavrentiev regularization.");
        parameter_handler.declare_entry
          ("filter_radius", "0.0", Patterns::Double(0.0), " Filter radius for the "
           "differential or post filter.");
        parameter_handler.declare_entry
          ("cutoff_n", "5", Patterns::Integer(0), " Cutoff parameter for the L2 "
           "projection filter.");
        parameter_handler.declare_entry
          ("filter_mean", "false", Patterns::Bool(), "Whether or not to filter the "
           "centered trajectory.");
      }
      parameter_handler.leave_subsection();

      parameter_handler.enter_subsection("ROM Configuration");
      {
        parameter_handler.declare_entry
          ("n_pod_dofs", "1", Patterns::Integer(1), "Number of POD vectors.");
        parameter_handler.declare_entry
          ("initial_time", "30.0", Patterns::Double(), " Initial time for the ROM.");
        parameter_handler.declare_entry
          ("final_time", "500.0", Patterns::Double(), " Final time for the ROM.");
        parameter_handler.declare_entry
          ("time_step", "1.0e-4", Patterns::Double(0.0), " Time step");
      }
      parameter_handler.leave_subsection();

      parameter_handler.enter_subsection("Output Configuration");
      {
        parameter_handler.declare_entry
          ("output_interval", "10", Patterns::Integer(0), " Number of iterations "
           "between which output is saved.");
      }
      parameter_handler.leave_subsection();
    }

    void Parameters::read_data(const std::string &file_name)
    {
      ParameterHandler parameter_handler;
      {
        std::ifstream file(file_name);
        configure_parameter_handler(parameter_handler);
        parameter_handler.read_input(file);
      }

      parameter_handler.enter_subsection("DNS Information");
      {
        reynolds_n = parameter_handler.get_double("reynolds_n");
      }
      parameter_handler.leave_subsection();

      parameter_handler.enter_subsection("Filtering Model Configuration");
      {
        auto filter_model_param = parameter_handler.get("filter_model");
        if (filter_model_param == std::string("Differential"))
          {
            filter_model = POD::FilterModel::Differential;
          }
        else if (filter_model_param == std::string("L2Projection"))
          {
            filter_model = POD::FilterModel::L2Projection;
          }
        else if (filter_model_param == std::string("PostDifferentialFilter"))
          {
            filter_model = POD::FilterModel::PostDifferentialFilter;
          }
        else if (filter_model_param == std::string("PostL2ProjectionFilter"))
          {
            filter_model = POD::FilterModel::PostL2ProjectionFilter;
          }
        else if (filter_model_param == std::string("LerayHybrid"))
          {
            filter_model = POD::FilterModel::LerayHybrid;
          }
        else if (filter_model_param == std::string("ADLavrentiev"))
          {
            filter_model = POD::FilterModel::ADLavrentiev;
          }
        else if (filter_model_param == std::string("ADTikonov"))
          {
            filter_model = POD::FilterModel::ADTikonov;
          }
        else
          {
            StandardExceptions::ExcNotImplemented();
          }

        noise_multiplier = parameter_handler.get_double("noise_multiplier");
        lavrentiev_parameter = parameter_handler.get_double("lavrentiev_parameter");
        filter_radius = parameter_handler.get_double("filter_radius");
        cutoff_n = parameter_handler.get_integer("cutoff_n");
        filter_mean = parameter_handler.get_bool("filter_mean");
      }
      parameter_handler.leave_subsection();

      parameter_handler.enter_subsection("ROM Configuration");
      {
        n_pod_dofs = parameter_handler.get_double("n_pod_dofs");
        initial_time = parameter_handler.get_double("initial_time");
        final_time = parameter_handler.get_double("final_time");
        time_step = parameter_handler.get_double("time_step");
      }
      parameter_handler.leave_subsection();

      parameter_handler.enter_subsection("Output Configuration");
      {
        output_interval = parameter_handler.get_integer("output_interval");
      }
      parameter_handler.leave_subsection();
    }
  }
}
