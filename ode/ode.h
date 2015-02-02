#ifndef __deal2_ode_pod_h
#define __deal2_ode_pod_h
#include <deal.II/lac/vector.h>

#include <iostream>
#include <memory>

using namespace dealii;
namespace ODE
{
  class OperatorBase
  {
  public:
    virtual void apply(Vector<double> &dst, const Vector<double> &src) = 0;
  };


  class RungeKuttaBase
  {
  public:
    RungeKuttaBase(std::unique_ptr<OperatorBase> rhs_function);
    virtual void step
    (double time_step, const Vector<double> &src, Vector<double> &dst) = 0;
  protected:
    std::unique_ptr<OperatorBase> rhs_function;
    unsigned int n_dofs;
  };


  class RungeKutta4 : public RungeKuttaBase
  {
  public:
    RungeKutta4(std::unique_ptr<OperatorBase> rhs_function);
    void step(double time_step, const Vector<double> &src,
              Vector<double> &dst) override;
  private:
    Vector<double> temp;
    Vector<double> step_1;
    Vector<double> step_2;
    Vector<double> step_3;
    Vector<double> step_4;
  };


  class RungeKutta4PostFilter : public RungeKutta4
  {
  public:
    RungeKutta4PostFilter
    (std::unique_ptr<OperatorBase> rhs_function,
     std::unique_ptr<OperatorBase> filter_function);
    void step
    (double time_step, const Vector<double> &src, Vector<double> &dst) override;
  protected:
    std::unique_ptr<OperatorBase> filter_function;
  };
}
#endif
