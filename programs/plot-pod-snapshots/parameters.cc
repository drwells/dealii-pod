#include <fstream>

#include "parameters.h"

namespace POD
{
  void Parameters::configure_parameter_handler
  (ParameterHandler &parameter_handler) const
  {
    parameter_handler.enter_subsection("DNS");
    {
      parameter_handler.declare_entry
        ("dimension", "2", Patterns::Integer(2), "Dimension (2 or 3) of the data.");
      parameter_handler.declare_entry
        ("fe_order", "2", Patterns::Integer(1), "Order of the finite element.");
      parameter_handler.declare_entry
        ("renumber", "false", Patterns::Bool(), "Whether or not to renumber "
         "the nodes with Cuthill-McKee.");
      parameter_handler.declare_entry
        ("triangulation_file_name", "triangulation.txt", Patterns::Anything(),
         "Name of the Triangulation file.");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("ROM");
    {
      parameter_handler.declare_entry
        ("pod_coefficients_file_name", "", Patterns::Anything(), "file containing POD coefficients.");
      parameter_handler.declare_entry
        ("start_time", "0.0", Patterns::Double(), "Start time of the ROM run.");
      parameter_handler.declare_entry
        ("time_step", "1e-4", Patterns::Double(0.0), "time step of the ROM run.");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Output");
    {
      parameter_handler.declare_entry
        ("output_interval", "10", Patterns::Integer(1), "How often to write a snapshot.");
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

    parameter_handler.enter_subsection("DNS");
    {
      dimension = parameter_handler.get_integer("dimension");
      fe_order = parameter_handler.get_integer("fe_order");
      renumber = parameter_handler.get_bool("renumber");
      triangulation_file_name = parameter_handler.get("triangulation_file_name");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("ROM");
    {
      pod_coefficients_file_name = parameter_handler.get("pod_coefficients_file_name");
      start_time = parameter_handler.get_double("start_time");
      time_step = parameter_handler.get_double("time_step");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Output");
    {
      output_interval = parameter_handler.get_integer("output_interval");
    }
    parameter_handler.leave_subsection();
  }
}
