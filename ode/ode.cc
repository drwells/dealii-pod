#include "ode.h"
namespace ODE
{
  RungeKuttaBase::RungeKuttaBase(std::unique_ptr<OperatorBase> rhs_function)
    : rhs_function {std::move(rhs_function)},
      n_dofs {numbers::invalid_unsigned_int}
  {}


  void EmptyOperator::apply(Vector<double> &dst, const Vector<double> &src)
  {
    StandardExceptions::ExcNotInitialized();
  }


  RungeKutta4::RungeKutta4()
    : RungeKuttaBase(std::unique_ptr<OperatorBase> (new EmptyOperator()))
  {}


  RungeKutta4::RungeKutta4(std::unique_ptr<OperatorBase> rhs_function)
    : RungeKuttaBase(std::move(rhs_function))
  {}


  void RungeKutta4::step
  (double time_step, const Vector<double> &src, Vector<double> &dst)
  {
    if (n_dofs == numbers::invalid_unsigned_int)
      {
        n_dofs = src.size();
        temp.reinit(n_dofs);
        step_1.reinit(n_dofs);
        step_2.reinit(n_dofs);
        step_3.reinit(n_dofs);
        step_4.reinit(n_dofs);
      }
    temp = src;
    rhs_function->apply(step_1, temp);
    temp = src;
    temp.add(0.5*time_step, step_1);
    rhs_function->apply(step_2, temp);
    temp = src;
    temp.add(0.5*time_step, step_2);
    rhs_function->apply(step_3, temp);
    temp = src;
    temp.add(time_step, step_3);
    rhs_function->apply(step_4, temp);

    dst = src;
    dst.add(time_step/6.0, step_1);
    dst.add(time_step/3.0, step_2);
    dst.add(time_step/3.0, step_3);
    dst.add(time_step/6.0, step_4);
  }


  RungeKutta4PostFilter::RungeKutta4PostFilter
  (std::unique_ptr<OperatorBase> rhs_function,
   std::unique_ptr<OperatorBase> filter_function)
    : RungeKutta4(std::move(rhs_function)),
      filter_function {std::move(filter_function)}
  {}


  void RungeKutta4PostFilter::step
  (double time_step, const Vector<double> &src, Vector<double> &dst)
  {
    Vector<double> temp(src.size());
    RungeKutta4::step(time_step, src, temp);
    filter_function->apply(dst, temp);
  }
}
