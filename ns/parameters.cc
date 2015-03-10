#include "parameters.h"

POD::NavierStokes::Parameters::Parameters()
{
  parameter_handler.enter_subsection("DNS Information");
  {
    parameter_handler.declare_entry
    ("reynolds_n", "50.0", Patterns::Double(0.0), " Reynolds number of the "
     "underlying simulation.");
    parameter_handler.declare_entry
    ("outflow_label", "3", Patterns::Integer(0), " Label of the outflow "
     "boundary.");
    parameter_handler.declare_entry
    ("fe_order", "2", Patterns::Integer(1), " Order of the finite element.");
    parameter_handler.declare_entry
    ("renumber", "false", Patterns::Bool(), " Whether or not to renumber "
     "the nodes with Cuthill-McKee.");
  }
  parameter_handler.leave_subsection();

  parameter_handler.enter_subsection("Filtering Model Configuration");
  {
    parameter_handler.declare_entry
    ("filter_model", "Differential",
     Patterns::Selection
     ("L2Projection|Differential|PostL2ProjectionFilter|PostDifferentialFilter"
      "|LerayHybrid"),
     " Filtering strategy for the POD-ROM.");
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
    ("patch_refinement", "2", Patterns::Integer(0), " Level of patch refinement"
     " for graphical output.");
    parameter_handler.declare_entry
    ("output_interval", "10", Patterns::Integer(0), " Number of iterations "
     "between which output is saved.");
    parameter_handler.declare_entry
    ("save_plot_pictures", "false", Patterns::Bool(), " Whether or not to save"
     " graphical output. This option is very expensive!");
  }
  parameter_handler.leave_subsection();
}

void POD::NavierStokes::Parameters::read_data(std::string file_name)
{
  {
    std::ifstream file(file_name);
    parameter_handler.read_input(file);
  }

  parameter_handler.enter_subsection("DNS Information");
  {
    reynolds_n = parameter_handler.get_double("reynolds_n");
    outflow_label = parameter_handler.get_integer("outflow_label");
    fe_order = parameter_handler.get_integer("fe_order");
    renumber = parameter_handler.get_bool("renumber");
  }
  parameter_handler.leave_subsection();

  parameter_handler.enter_subsection("Filtering Model Configuration");
  {
    if (parameter_handler.get("filter_model") == std::string("Differential"))
      {
        filter_model = POD::FilterModel::Differential;
      }
    else if (parameter_handler.get("filter_model") == std::string("L2Projection"))
      {
        filter_model = POD::FilterModel::L2Projection;
      }
    else if (parameter_handler.get("filter_model") ==
             std::string("PostDifferentialFilter"))
      {
        filter_model = POD::FilterModel::PostDifferentialFilter;
      }
    else if (parameter_handler.get("filter_model") ==
             std::string("PostL2ProjectionFilter"))
      {
        filter_model = POD::FilterModel::PostL2ProjectionFilter;
      }
    else if (parameter_handler.get("filter_model") == std::string("LerayHybrid"))
      {
        filter_model = POD::FilterModel::LerayHybrid;
      }
    else
      {
        StandardExceptions::ExcNotImplemented();
      }

    filter_radius = parameter_handler.get_double("filter_radius");
    cutoff_n = parameter_handler.get_integer("cutoff_n");
    filter_mean = parameter_handler.get_bool("filter_mean");
  }
  parameter_handler.leave_subsection();

  parameter_handler.enter_subsection("ROM Configuration");
  {
    initial_time = parameter_handler.get_double("initial_time");
    final_time = parameter_handler.get_double("final_time");
    time_step = parameter_handler.get_double("time_step");
  }
  parameter_handler.leave_subsection();

  parameter_handler.enter_subsection("Output Configuration");
  {
    patch_refinement = parameter_handler.get_integer("patch_refinement");
    output_interval = parameter_handler.get_integer("output_interval");
    save_plot_pictures = parameter_handler.get_bool("save_plot_pictures");
  }
  parameter_handler.leave_subsection();
}
