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
        ("snapshot_glob", "snapshot-*h5", Patterns::Anything(), "Glob to match for "
         "the snapshot files.");
      parameter_handler.declare_entry
        ("triangulation_file_name", "triangulation.txt", Patterns::Anything(),
         "Name of the Triangulation file.");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("ROM");
    {
      parameter_handler.declare_entry
        ("n_pod_vectors", "100", Patterns::Integer(1), "Number of POD vectors to "
         "save.");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Output");
    {
      parameter_handler.declare_entry
        ("save_plot_pictures", "false", Patterns::Bool(), " Whether or not to save"
         " graphical output.");
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

    parameter_handler.enter_subsection("DNS");
    {
      dimension = parameter_handler.get_integer("dimension");
      fe_order = parameter_handler.get_integer("fe_order");
      renumber = parameter_handler.get_bool("renumber");
      snapshot_glob = parameter_handler.get("snapshot_glob");
      triangulation_file_name = parameter_handler.get("triangulation_file_name");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("ROM");
    {
      n_pod_vectors = parameter_handler.get_integer("n_pod_vectors");
    }
    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Output");
    {
      save_plot_pictures = parameter_handler.get_bool("save_plot_pictures");
    }
    parameter_handler.leave_subsection();
  }
}
