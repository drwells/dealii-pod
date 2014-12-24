#ifndef __deal2_ode_pod_h
#define __deal2_ode_pod_h
#include <deal.II/lac/vector.h>

#include <iostream>
#include <memory>

using namespace dealii;
namespace ODE
{
  class NonlinearOperatorBase
  {
  public:
    NonlinearOperatorBase();
    virtual void apply(Vector<double> &dst, const Vector<double> &src);
  };

  class RungeKuttaBase
  {
  public:
    RungeKuttaBase();
    virtual void step(double time_step, const Vector<double> &src,
                      Vector<double> &dst);
  protected:
    // The pointer is needed to get around the "object slicing" problem. See
    // the wikipedia entry.
    std::unique_ptr<NonlinearOperatorBase> rhs_function;
    unsigned int n_dofs;
  };


  class RungeKutta4 : public RungeKuttaBase
  {
  public:
    RungeKutta4(std::unique_ptr<NonlinearOperatorBase> rhs_function);
    void step(double time_step, const Vector<double> &src,
              Vector<double> &dst) override;
  private:
    Vector<double> temp;
    Vector<double> step_1;
    Vector<double> step_2;
    Vector<double> step_3;
    Vector<double> step_4;
  };
}
#endif