/* ---------------------------------------------------------------------
 * Copyright (C) 2014-2015 David Wells
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
#ifndef dealii__rom_h5_block_h
#define dealii__rom_h5_block_h

#include <deal.II/lac/block_vector.h>

#include <hdf5.h>

#include <string>
#include <vector>

namespace H5
{
  template<typename T>
  void load_block_vector(std::string file_name,
                         dealii::BlockVector<T> &block_vector);

  template<typename T>
  void save_block_vector(std::string file_name,
                         dealii::BlockVector<T> &block_vector);

  template<typename T>
  void load_full_matrix(std::string file_name,
                        T &matrix);

  template<typename T>
  void save_full_matrix(std::string file_name,
                        T &matrix);

  template<typename T>
  void load_vector(const std::string &file_name,
                   T &vector);

  template<typename T>
  void save_vector(const std::string &file_name,
                   T &vector);

  template<typename T>
  void load_full_matrices(const std::string &file_name,
                          std::vector<T> &matrices);

  template<typename T>
  void save_full_matrices(const std::string &file_name,
                          const std::vector<T> &matrices);
}
#endif
