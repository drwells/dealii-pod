#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <deal.II-pod/extra/extra.h>
#include <deal.II-pod/pod/pod.h>
#include <deal.II-pod/h5/h5.h>

#include "parameters.h"


namespace POD
{
  using namespace dealii;

  template<int dim>
  class PlotPODSnapshots
  {
  public:
    PlotPODSnapshots(const Parameters &parameters);
    void run();
  private:
    const Parameters                                  parameters;
    Triangulation<dim>                                triangulation;
    const FE_Q<dim>                                   fe;
    std::shared_ptr<DoFHandler<dim>>                  dof_handler;
    std::shared_ptr<BlockVector<double>>              mean_vector;
    std::shared_ptr<std::vector<BlockVector<double>>> pod_vectors;

    void load_mesh();
    void save_pod_snapshots();
  };


  template<int dim>
  PlotPODSnapshots<dim>::PlotPODSnapshots(const Parameters &parameters)
    :
    parameters(parameters),
    fe(parameters.fe_order),
    dof_handler {std::make_shared<DoFHandler<dim>>()},
    mean_vector {std::make_shared<BlockVector<double>>()},
    pod_vectors {std::make_shared<std::vector<BlockVector<double>>>()}
  {}


  template<int dim>
  void PlotPODSnapshots<dim>::load_mesh()
  {
    std::ifstream in_stream(parameters.triangulation_file_name);
    boost::archive::text_iarchive archive(in_stream);
    archive >> triangulation;

    dof_handler->initialize(triangulation, fe);

    if (parameters.renumber)
      {
        DoFRenumbering::boost::Cuthill_McKee(*dof_handler);
      }
  }


  template<int dim>
  void PlotPODSnapshots<dim>::save_pod_snapshots()
  {
    load_pod_basis("pod-vector*h5", "mean-vector.h5", *mean_vector, *pod_vectors);
    FullMatrix<double> pod_coefficients;
    H5::load_full_matrix(parameters.pod_coefficients_file_name, pod_coefficients);
    AssertThrow(pod_vectors->size() >= pod_coefficients.n(), ExcMessage
                ("The number of POD vectors and number of columns in the data "
                 "matrix should agree."));
    if (pod_vectors->size() > pod_coefficients.n())
      {
        pod_vectors->resize(pod_coefficients.n());
      }

    PODOutput<dim> pod_output(dof_handler, mean_vector, pod_vectors, "pod-snapshot-");
    Vector<double> solution(pod_coefficients.n());

    double time = parameters.start_time;
    for (unsigned int i = 0; i < pod_coefficients.m(); i+=parameters.output_interval)
      {
        for (unsigned int j = 0; j < pod_coefficients.n(); ++j)
          {
            solution[j] = pod_coefficients(i, j);
          }

        pod_output.save_solution(solution, time, i);
        time += parameters.time_step*parameters.output_interval;
      }
  }


  template<int dim>
  void PlotPODSnapshots<dim>::run()
  {
    load_mesh();
    save_pod_snapshots();
  }
}


int main(int argc, char **argv)
{
  using namespace POD;
  Utilities::MPI::MPI_InitFinalize mpi_initialization
  (argc, argv, numbers::invalid_unsigned_int);
  {
    POD::Parameters parameters;
    parameters.read_data("parameter-file.prm");
    if (parameters.dimension == 2)
      {
        PlotPODSnapshots<2> pod_vectors(parameters);
        pod_vectors.run();
      }
    else
      {
        PlotPODSnapshots<3> pod_vectors(parameters);
        pod_vectors.run();
      }
  }
}
