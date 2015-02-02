#include "ode.h"
namespace ODE
{
  // These stubs are necessary because I pass a not-quite abstract type around.
  // I should probably rewrite this to use templates because I want incomplete
  // types.
  {}


  RungeKuttaBase::RungeKuttaBase()
  {}

  void RungeKuttaBase::step(double time_step,
                            const Vector<double> &src,
                            Vector<double> &dst)
  {
    std::cerr << "this method is a stub and does nothing!" << std::endl;
  }

  RungeKutta4::RungeKutta4(std::unique_ptr<NonlinearOperatorBase> rhs_function)
    : RungeKuttaBase()
  {
    this->rhs_function = std::move(rhs_function);
    n_dofs = numbers::invalid_unsigned_int;
  }

  void RungeKutta4::step(double time_step, const Vector<double> &src, Vector<double> &dst)
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
}