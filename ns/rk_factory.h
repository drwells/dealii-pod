/* ---------------------------------------------------------------------
 * Copyright (C) 2014 David Wells
 *
 * This file is NOT part of the deal.II library.
 *
 * This file is free software; you can use it, redistribute it, and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 * This program is based on step-26 of the deal.ii library.
 *
 * Author: David Wells, Virginia Tech, 2014
 */
#ifndef dealii__rom_rk_factory_h
#define dealii__rom_rk_factory_h
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "../ode/ode.h"
#include "ns.h"
#include "parameters.h"

namespace POD
{
  namespace NavierStokes
  {
    using namespace dealii;
    std::pair<std::string, std::unique_ptr<ODE::RungeKuttaBase>> rk_factory
    (const FullMatrix<double>              &boundary_matrix,
     const FullMatrix<double>              &joint_convection,
     const FullMatrix<double>              &laplace_matrix,
     const FullMatrix<double>              &linear_operator,
     const Vector<double>                  &mean_contribution_vector,
     const FullMatrix<double>              &mass_matrix,
     const std::vector<FullMatrix<double>> &nonlinear_operator,
     const POD::NavierStokes::Parameters   &parameters);
  }
}
#endif
