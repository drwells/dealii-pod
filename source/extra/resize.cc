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
#include <deal.II-pod/extra/resize.h>

namespace POD
{
  using namespace dealii;

  namespace extra
  {
    void resize(FullMatrix<double> &matrix,
                const unsigned int new_size)
    {
      FullMatrix<double> temp;
      temp = matrix;

      // free all memory, then reallocate
      TableIndices<2> empty_table(0, 0);
      matrix.reinit(empty_table);
      TableIndices<2> indices(new_size, new_size);
      matrix.reinit(indices);
      for (unsigned int i = 0; i < new_size; ++i)
        {
          for (unsigned int j = 0; j < new_size; ++j)
            {
              matrix(i, j) = temp(i, j);
            }
        }
    }

    void resize(Vector<double> &vector,
                const unsigned int new_size)
    {
      Vector<double> temp;
      temp = vector;

      // free all memory, then reallocate
      vector.reinit(0);
      vector.reinit(new_size);
      for (unsigned int i = 0; i < new_size; ++i)
        {
          vector[i] = temp[i];
        }
    }
  }
}
