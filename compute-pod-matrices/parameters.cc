#include "parameters.h"

namespace ComputePOD
{
  void Parameters::configure_parameter_handler
  (ParameterHandler &parameter_handler) const
  {
    parameter_handler.enter_subsection("DNS");
    {
      parameter_handler.declare_entry
        ("fe_order", "2", Patterns::Integer(1), "Order of the finite element.");
      parameter_handler.declare_entry
        ("renumber", "false", Patterns::Bool(), "Whether or not to renumber "
         "the nodes with Cuthill-McKee.");
      parameter_handler.declare_entry
        ("triangulation_file_name", "triangulation.txt", Patterns::Anything(),
         "Name of the Triangulation file.");
      parameter_handler.declare_entry
        ("reynolds_n", "1.0", Patterns::Double(), "Reynolds number.");
      parameter_handler.declare_entry
        ("outflow_label", "10", Patterns::Integer(1), "Outflow label.");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("ROM");
    {
      parameter_handler.declare_entry
        ("center_trajectory", "true", Patterns::Bool(), "Whether or not to center "
         "the trajectory.");
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
      fe_order = parameter_handler.get_integer("fe_order");
      renumber = parameter_handler.get_bool("renumber");
      triangulation_file_name = parameter_handler.get("triangulation_file_name");
      reynolds_n = parameter_handler.get_double("reynolds_n");
      outflow_label = parameter_handler.get_integer("outflow_label");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("ROM");
    {
      center_trajectory = parameter_handler.get_bool("center_trajectory");
    }
    parameter_handler.leave_subsection();
  }
}
