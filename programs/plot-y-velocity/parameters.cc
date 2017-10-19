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
      ("snapshot_glob", "snapshot-*h5", Patterns::Anything(), "Glob to match for "
       "the snapshot files.");
    parameter_handler.declare_entry
      ("triangulation_file_name", "triangulation.txt", Patterns::Anything(),
       "Name of the Triangulation file.");
    parameter_handler.declare_entry
      ("time_step", "0.01", Patterns::Double(),
       "Time distance between snapshots.");
  }
  parameter_handler.leave_subsection();

  parameter_handler.enter_subsection("Output Configuration");
  {
    parameter_handler.declare_entry
    ("patch_level", "2", Patterns::Integer(0),
     "Patch refinement level.");
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
    snapshot_glob = parameter_handler.get("snapshot_glob");
    triangulation_file_name = parameter_handler.get("triangulation_file_name");
    time_step = parameter_handler.get_double("time_step");
  }
  parameter_handler.leave_subsection();

  parameter_handler.enter_subsection("Output Configuration");
  {
    patch_level = parameter_handler.get_integer("patch_level");
  }
  parameter_handler.leave_subsection();
}
