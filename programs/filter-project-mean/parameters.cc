#include "parameters.h"

void Parameters::configure_parameter_handler
(ParameterHandler &parameter_handler) const
{
  parameter_handler.enter_subsection("DNS Information");
  {
    parameter_handler.declare_entry
      ("fe_order", "2", Patterns::Integer(1), "Order of the finite element.");
    parameter_handler.declare_entry
      ("renumber", "false", Patterns::Bool(), "Whether or not to renumber "
       "the nodes with Cuthill-McKee.");
    parameter_handler.declare_entry
    ("outflow_label", "3", Patterns::Integer(0), " Label of the outflow "
     "boundary.");
    parameter_handler.declare_entry
      ("triangulation_file_name", "triangulation.txt", Patterns::Anything(),
       "Name of the Triangulation file.");
  }
  parameter_handler.leave_subsection();

  parameter_handler.enter_subsection("ROM Configuration");
  {
    parameter_handler.declare_entry
      ("pod_vector_glob", "pod_vector*h5", Patterns::Anything(), "Glob to "
       "match for the POD vector files.");
    parameter_handler.declare_entry
      ("mean_vector_file_name", "mean_vector*h5", Patterns::Anything(), "Glob to "
       "match for the mean vector.");
    parameter_handler.declare_entry
    ("filter_radius", "0.0", Patterns::Double(0.0), " Filter radius for the "
     "differential or post filter.");
  }
  parameter_handler.leave_subsection();
}

void Parameters::read_data(const std::string &file_name)
{
  ParameterHandler parameter_handler;
  {
    std::ifstream file(file_name);
    configure_parameter_handler(parameter_handler);
    parameter_handler.parse_input(file);
  }

  parameter_handler.enter_subsection("DNS Information");
  {
    fe_order = parameter_handler.get_integer("fe_order");
    renumber = parameter_handler.get_bool("renumber");
    outflow_label = parameter_handler.get_integer("outflow_label");
    triangulation_file_name = parameter_handler.get("triangulation_file_name");
  }
  parameter_handler.leave_subsection();

  parameter_handler.enter_subsection("ROM Configuration");
  {
    pod_vector_glob = parameter_handler.get("pod_vector_glob");
    mean_vector_file_name = parameter_handler.get("mean_vector_file_name");
    filter_radius = parameter_handler.get_double("filter_radius");
  }
  parameter_handler.leave_subsection();
}
