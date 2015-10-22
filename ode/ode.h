/* ---------------------------------------------------------------------
 * Copyright (C) 2014 David Wells
 *
 * This file is NOT part of the deal.II library.
 *
 * This file is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 *
 * Author: David Wells, Virginia Tech, 2014-2015;
 *         David Wells, Rensselaer Polytechnic Institute, 2015
 */
#ifndef dealii__rom_ode_pod_h
#define dealii__rom_ode_pod_h
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
    virtual ~OperatorBase() = default;
  };

  // Empty object, so that we may instantiate things without null pointers. It
  // is not quite a null object since EmptyOperator::apply throws an exception.
  class EmptyOperator : public OperatorBase
  {
    virtual void apply(Vector<double> &dst, const Vector<double> &src) override;
  };


  class RungeKuttaBase
  {
  public:
    RungeKuttaBase(std::unique_ptr<OperatorBase> rhs_function);
    virtual void step
    (double time_step, const Vector<double> &src, Vector<double> &dst) = 0;
    virtual ~RungeKuttaBase() = default;
  protected:
    std::unique_ptr<OperatorBase> rhs_function;
    unsigned int n_dofs;
  };


  class RungeKutta4 : public RungeKuttaBase
  {
  public:
    RungeKutta4();
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
