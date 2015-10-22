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

#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <hdf5.h>

namespace H5
{
  template<typename T>
  void load_block_vector(std::string file_name,
                         dealii::BlockVector<T> &block_vector)
  {
    hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    std::vector<hsize_t> n_obj(1);
    H5Gget_num_objs(file_id, n_obj.data());
    hsize_t n_blocks = n_obj[0];
    block_vector.reinit(n_blocks);
    block_vector.collect_sizes();

    for (unsigned int i = 0; i < n_blocks; ++i)
      {
        std::string dataset_name = "/a" + dealii::Utilities::int_to_string(i);
        hid_t dataset = H5Dopen1(file_id, dataset_name.c_str());
        hid_t datatype = H5Dget_type(dataset);
        hid_t dataspace = H5Dget_space(dataset);
        int rank = H5Sget_simple_extent_ndims(dataspace);
        Assert(rank == 1, dealii::ExcInternalError());

        std::vector<hsize_t> dims(rank);
        std::vector<hsize_t> max_dims(rank);
        H5Sget_simple_extent_dims(dataspace, dims.data(), max_dims.data());
        hsize_t bufsize = H5Dget_storage_size(dataset);
        block_vector.block(i).reinit(bufsize/sizeof(T));
        H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                static_cast<void *>(&(block_vector.block(i)[0])));
        H5Sclose(dataspace);
        H5Tclose(datatype);
        H5Dclose(dataset);
      }

    H5Fclose(file_id);
  }

  // TODO it should be possible to parameterize this by type. However, I do not
  // see an easy way to go from BlockVector<double> to H5T_NATIVE_DOUBLE.
  template<typename T>
  void save_block_vector(std::string file_name,
                         dealii::BlockVector<T> &block_vector)
  // Save a deal.II block vector to an HDF5 file as components a0, a1, etc.
  {
    hid_t file_id = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                              H5P_DEFAULT);
    for (unsigned int i = 0; i < block_vector.n_blocks(); ++i)
      {
        hsize_t n_dofs[1];
        n_dofs[0] = block_vector.block(i).size();
        hid_t dataspace_id = H5Screate_simple(1, n_dofs, nullptr);
        std::string dataset_name = "/a" + dealii::Utilities::int_to_string(i);
        hid_t dataset_id = H5Dcreate2 (file_id, dataset_name.c_str (),
                                       H5T_NATIVE_DOUBLE, dataspace_id,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 &(block_vector.block(i)[0]));
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
      }
    H5Fclose(file_id);
  }

  // TODO it should be possible to parameterize this by type.
  template<typename T>
  void load_full_matrix(std::string file_name, T &matrix)
  // load a deal.II full matrix.
  {
    hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    std::string dataset_name = "/a";
    hid_t dataset = H5Dopen1(file_id, dataset_name.c_str());
    hid_t datatype = H5Dget_type(dataset);
    hid_t dataspace = H5Dget_space(dataset);
    int rank = H5Sget_simple_extent_ndims(dataspace);
    Assert(rank == 2, dealii::ExcInternalError());

    std::vector<hsize_t> dims(rank);
    std::vector<hsize_t> max_dims(rank);
    H5Sget_simple_extent_dims(dataspace, dims.data(), max_dims.data());
    hsize_t bufsize = H5Dget_storage_size(dataset);
    matrix.reinit(dims[0], dims[1]);
    H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            static_cast<void *>(&(matrix(0, 0))));

    H5Sclose(dataspace);
    H5Tclose(datatype);
    H5Dclose(dataset);
    H5Fclose(file_id);
  }

  // TODO it should be possible to parameterize this by type.
  template<typename T>
  void save_full_matrix(std::string file_name, T &matrix)
  // Save a deal.II full matrix.
  {
    hid_t file_id = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                              H5P_DEFAULT);
    hsize_t dims[2];

    dims[0] = matrix.m();
    dims[1] = matrix.n();
    hid_t dataspace_id = H5Screate_simple(2, dims, nullptr);
    std::string dataset_name = "/a";
    hid_t dataset_id = H5Dcreate2(file_id, dataset_name.c_str(),
                                  H5T_NATIVE_DOUBLE, dataspace_id,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             &matrix(0, 0));
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
  }



  template<typename T>
  void load_vector(const std::string &file_name, const T &vector)
  {
    hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    std::string dataset_name = "/a";
    hid_t dataset = H5Dopen1(file_id, dataset_name.c_str());
    // TODO assert that this is double precision
    hid_t datatype = H5Dget_type(dataset);
    hid_t dataspace = H5Dget_space(dataset);
    int rank = H5Sget_simple_extent_ndims(dataspace);
    Assert(rank == 1, dealii::ExcInternalError());

    // Since rank must be 1...
    hsize_t dims[1];
    hsize_t max_dims[1];
    H5Sget_simple_extent_dims(dataspace, dims, max_dims);
    hsize_t bufsize = H5Dget_storage_size(dataset);
    vector.reinit(dims[0]);
    H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            static_cast<void *>(&(vector[0])));

    H5Sclose(dataspace);
    H5Tclose(datatype);
    H5Dclose(dataset);
    H5Fclose(file_id);
  }



  template<typename T>
  void save_vector(const std::string &file_name, T &vector)
  {
    hid_t file_id = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                              H5P_DEFAULT);

    hsize_t n_dofs[1];
    n_dofs[0] = vector.size();
    hid_t dataspace_id = H5Screate_simple(1, n_dofs, nullptr);
    std::string dataset_name {"/a"};
    hid_t dataset_id = H5Dcreate2 (file_id, dataset_name.c_str(),
                                   H5T_NATIVE_DOUBLE, dataspace_id,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             &(vector[0]));
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
  }



  template<typename T>
  void load_full_matrices(const std::string &file_name,
                          std::vector<T> &matrices)
  {
    hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    hsize_t n_obj;
    H5Gget_num_objs(file_id, &n_obj);
    matrices.resize(n_obj);

    for (unsigned int i = 0; i < n_obj; ++i)
      {
        std::string dataset_name = "/a" + dealii::Utilities::int_to_string(i);
        hid_t dataset = H5Dopen1(file_id, dataset_name.c_str());
        hid_t datatype = H5Dget_type(dataset);
        hid_t dataspace = H5Dget_space(dataset);
        int rank = H5Sget_simple_extent_ndims(dataspace);
        Assert(rank == 2, dealii::ExcInternalError());

        hsize_t dims[2];
        hsize_t max_dims[2];
        H5Sget_simple_extent_dims(dataspace, dims, max_dims);
        matrices.at(i).reinit(dims[0], dims[1]);
        H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                static_cast<void *>(&(matrices.at(i)(0, 0))));
        H5Sclose(dataspace);
        H5Tclose(datatype);
        H5Dclose(dataset);
      }

    H5Fclose(file_id);
  }



  template<typename T>
  void save_full_matrices(const std::string &file_name,
                          const std::vector<T> &matrices)
  {
    hid_t file_id = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                              H5P_DEFAULT);
    for (unsigned int i = 0; i < matrices.size(); ++i)
      {
        hsize_t dims[2];
        dims[0] = matrices[i].m();
        dims[1] = matrices[i].n();
        hid_t dataspace_id = H5Screate_simple(2, dims, nullptr);
        std::string dataset_name = "/a" + dealii::Utilities::int_to_string(i);
        hid_t dataset_id = H5Dcreate2(file_id, dataset_name.c_str(),
                                      H5T_NATIVE_DOUBLE, dataspace_id,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 &(matrices[i](0, 0)));
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
      }
    H5Fclose(file_id);
  }
}
#endif
