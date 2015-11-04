/* ---------------------------------------------------------------------
 * Copyright (C) 2014-2015 David Wells
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
 * Author: David Wells, Rensselaer Polytechnic Institute, 2015
 */
#include <deal.II/base/table_indices.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>

namespace POD
{
  using namespace dealii;

  namespace extra
  {
    /*
     * Resize a square matrix to be new_size x new_size.
     */
    void resize(FullMatrix<double> &matrix,
                const unsigned int new_size);

    /*
     * Resize a vector to have length new_size.
     */
    void resize(Vector<double> &vector,
                const unsigned int new_size);
  }
}
