#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/petsc_matrix_free.h>
#include <deal.II/lac/petsc_vector_base.h>
#include <deal.II/lac/petsc_full_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/slepc_solver.h>

#include <glob.h>

namespace POD
{
  using namespace dealii::PETScWrappers;
  void copy_vector(const VectorBase &src, unsigned int start_src,
                   unsigned int start_dst, unsigned int total, VectorBase &dst)
  {
      for (unsigned int index = 0; index < total; ++index)
      {
          dst[start_dst + index] = src[start_src + index];
      }
  }

  class MethodOfSnapshots : public MatrixFree
  {
  public:
    MethodOfSnapshots(SparseMatrix &mass_matrix, FullMatrix &snapshots,
                      unsigned int num_blocks)
      : num_blocks (num_blocks), mass_matrix (mass_matrix), snapshots (snapshots)
    {
// use the reinit function to set the number of rows and columns. This is,
// AFAIK, the only way to set the underlying PETSc matrix representation's
// values.
      reinit(this->mass_matrix.m(), this->mass_matrix.n(),
      this->mass_matrix.m(), this->mass_matrix.n());
    }

    void vmult(VectorBase &dst, const VectorBase &src) const
    {
      Vector tmp_component_src(src.size()/num_blocks);
      Vector tmp_component_dst(src.size()/num_blocks);
      Vector tmp_short(snapshots.m());
      Vector tmp_long(snapshots.n());

      tmp_component_src.compress(dealii::VectorOperation::add);
      tmp_component_dst.compress(dealii::VectorOperation::add);
      tmp_short.compress(dealii::VectorOperation::add);
      tmp_long.compress(dealii::VectorOperation::add);

// TODO this requires *a lot* of extra copying. There may be a better way to do
// this with VectorView.
      for (unsigned int j = 0; j < num_blocks; ++j)
      {
        copy_vector(src, 0, 0, src.size()/num_blocks, tmp_component_src);
        mass_matrix.vmult(tmp_component_dst, tmp_component_src);
        copy_vector(tmp_component_src, 0, j*tmp_long.size()/num_blocks,
                    tmp_component_src.size(), tmp_long);
      }
      snapshots.vmult(tmp_short, tmp_long);
      snapshots.Tvmult(dst, tmp_short);
    }

    void vmult_add(VectorBase &dst, const VectorBase &src) const
    {
      Assert(false, dealii::ExcNotImplemented());
      Vector tmp_short(snapshots.m());
      Vector tmp_long(snapshots.n());

      tmp_short.compress(dealii::VectorOperation::add);
      tmp_long.compress(dealii::VectorOperation::add);

      mass_matrix.vmult(tmp_long, src);
      snapshots.vmult(tmp_short, tmp_long);
      snapshots.Tvmult_add(dst, tmp_short);
    }

    void Tvmult(VectorBase &dst, const VectorBase &src) const
    {
      Assert(false, dealii::ExcNotImplemented());
      Vector tmp_short(snapshots.m());
      Vector tmp_long(snapshots.n());

      tmp_short.compress(dealii::VectorOperation::add);
      tmp_long.compress(dealii::VectorOperation::add);

      snapshots.vmult(tmp_short, src);
      snapshots.Tvmult(tmp_long, tmp_short);
      mass_matrix.vmult(dst, tmp_long);
    }

    void Tvmult_add(VectorBase &dst, const VectorBase &src) const
    {
      Assert(false, dealii::ExcNotImplemented());
      Vector tmp_short(snapshots.m());
      Vector tmp_long(snapshots.n());

      tmp_short.compress(dealii::VectorOperation::add);
      tmp_long.compress(dealii::VectorOperation::add);

      snapshots.vmult(tmp_short, src);
      snapshots.Tvmult(tmp_long, tmp_short);
      mass_matrix.Tvmult_add(dst, tmp_long);
    }
    unsigned int num_blocks;

  private:
    SparseMatrix &mass_matrix;
    FullMatrix &snapshots;
  };

  class PODBasis
  {
  public:
    std::map<int, dealii::Vector<double>> vectors;
    std::vector<double> singular_values;
    unsigned int get_num_pod_vectors() const;
    void project_load_vector(dealii::Vector<double> &load_vector,
                             dealii::Vector<double> &pod_load_vector);
    void project_to_fe(const dealii::Vector<double> &pod_vector,
                       dealii::Vector<double> &fe_vector) const;
  };

  unsigned int PODBasis::get_num_pod_vectors() const
  {
    return singular_values.size();
  }

  void PODBasis::project_load_vector(dealii::Vector<double> &load_vector,
                                     dealii::Vector<double> &pod_load_vector)
  {
    pod_load_vector.reinit(get_num_pod_vectors());
    for (unsigned int j = 0; j < get_num_pod_vectors(); ++j)
      {
        pod_load_vector[j] = vectors[j] * load_vector;
      }
  }

  void PODBasis::project_to_fe(const dealii::Vector<double> &pod_vector,
                               dealii::Vector<double> &fe_vector) const
  {
    fe_vector.reinit(vectors.at(0).size());
    for (unsigned int j = 0; j < this->get_num_pod_vectors(); ++j)
      {
        fe_vector.add(pod_vector[j], vectors.at(j));
      }
  }

  void pod_basis(dealii::SparseMatrix<double> &mass_matrix,
                 std::map<int, dealii::Vector<double>> &snapshots,
                 const unsigned int num_pod_vectors,
                 PODBasis &result)
  {
    Assert(num_pod_vectors > 0, dealii::ExcInternalError());

    auto &sparsity_pattern = mass_matrix.get_sparsity_pattern();
    SparseMatrix petsc_mass_matrix(sparsity_pattern);
    for (dealii::types::global_dof_index row = 0; row < mass_matrix.m(); ++row)
      {
        for (auto row_entry = sparsity_pattern.begin(row);
             row_entry != sparsity_pattern.end(row);
             ++row_entry)
          {
            petsc_mass_matrix.set(row, row_entry->column(),
                                  mass_matrix(row, row_entry->column()));
          }
      }
    petsc_mass_matrix.compress(dealii::VectorOperation::add);

    // TODO get rid of hardcoded vector size
    unsigned int num_blocks = 3;
    FullMatrix snapshot_matrix(snapshots.size(), num_blocks*mass_matrix.m());
    std::vector<unsigned int> column_indices;
    for (dealii::types::global_dof_index j = 0; j < mass_matrix.m(); ++j)
      {
        column_indices.push_back(j);
      }
    for (dealii::types::global_dof_index j = 0; j < snapshots.size(); ++j)
      {
        auto snapshot = snapshots[static_cast<int>(j)];
        std::vector<double> snapshot_vector(column_indices.size());
        for (unsigned int column_index = 0;
             column_index < column_indices.size();
             ++column_index)
          {
            snapshot_vector[column_index] = snapshot[column_index];
          }
        snapshot_matrix.set(static_cast<MatrixBase::value_type>(j),
                            column_indices,
                            snapshot_vector);
      }
    snapshot_matrix.compress(dealii::VectorOperation::add);

    MethodOfSnapshots pod_data(petsc_mass_matrix, snapshot_matrix, num_blocks);
    dealii::SolverControl solver_control (mass_matrix.m(), 1e-9);
    dealii::SLEPcWrappers::SolverArnoldi eigensolver (solver_control);
    eigensolver.set_which_eigenpairs(EPS_LARGEST_MAGNITUDE);

    std::vector<Vector> eigenvectors;
    for (unsigned int i = 0; i < num_pod_vectors; ++i)
      {
        eigenvectors.push_back(Vector(pod_data.m()));
      }
    eigensolver.solve(pod_data, result.singular_values, eigenvectors,
                      num_pod_vectors);

    std::map<int, std::vector<double>> pod_vectors;
    for (unsigned int i = 0; i < num_pod_vectors; ++i)
      {
        dealii::Vector<double> outvalues(eigenvectors[0].size());
// TODO this is an icky bit of copying, but there does not seem to be a built
// in for it.
        for (unsigned int j = 0; j < eigenvectors[i].size(); ++j)
          {
            outvalues[j] = eigenvectors[i][j];
          }
        result.vectors.insert(std::pair<int, dealii::Vector<double>>(i, outvalues));
      }
  }

  void reduced_matrix(dealii::SparseMatrix<double> &mass_matrix,
                      PODBasis &pod_basis, dealii::FullMatrix<double> &pod_mass_matrix)
  // Compute the reduced matrix from a given POD basis.
  {
    auto num_pod_vectors = pod_basis.get_num_pod_vectors();
    pod_mass_matrix.reinit(num_pod_vectors, num_pod_vectors);

    dealii::Vector<double> tmp(mass_matrix.m());
    for (unsigned int i = 0; i < num_pod_vectors; ++i)
      {
        for (unsigned int j = 0; j < num_pod_vectors; ++j)
          {
            mass_matrix.vmult(tmp, pod_basis.vectors[j]);
            pod_mass_matrix(i, j) = tmp * pod_basis.vectors[i];
          }
      }
  }
}
