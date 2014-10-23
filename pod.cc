#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/petsc_matrix_free.h>
#include <deal.II/lac/petsc_vector_base.h>
#include <deal.II/lac/petsc_full_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/slepc_solver.h>

namespace POD
{
  using namespace dealii::PETScWrappers;
  class MethodOfSnapshots : public MatrixFree
  {
  public:
    MethodOfSnapshots(SparseMatrix &mass_matrix, FullMatrix &snapshots)
      : mass_matrix {mass_matrix}, snapshots {snapshots}
    {
// use the reinit function to set the number of rows and columns. This is,
//AFAIK, the only way to set the underlying PETSc matrix representation's
// values.
      reinit(this->mass_matrix.m(), this->mass_matrix.n(),
      this->mass_matrix.m(), this->mass_matrix.n());
    }

    void vmult(VectorBase &dst, const VectorBase &src) const
    {
      Vector tmp_short(snapshots.m());
      Vector tmp_long(snapshots.n());

      tmp_short.compress(dealii::VectorOperation::add);
      tmp_long.compress(dealii::VectorOperation::add);

      mass_matrix.vmult(tmp_long, src);
      snapshots.vmult(tmp_short, tmp_long);
      snapshots.Tvmult(dst, tmp_short);
    }

    void vmult_add(VectorBase &dst, const VectorBase &src) const
    {
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
      Vector tmp_short(snapshots.m());
      Vector tmp_long(snapshots.n());

      tmp_short.compress(dealii::VectorOperation::add);
      tmp_long.compress(dealii::VectorOperation::add);

// TODO I haven't really checked to see if this works.
      snapshots.vmult(tmp_short, src);
      snapshots.Tvmult(tmp_long, tmp_short);
      mass_matrix.vmult(dst, tmp_long);
    }

    void Tvmult_add(VectorBase &dst, const VectorBase &src) const
    {
      Vector tmp_short(snapshots.m());
      Vector tmp_long(snapshots.n());

      tmp_short.compress(dealii::VectorOperation::add);
      tmp_long.compress(dealii::VectorOperation::add);

      snapshots.vmult(tmp_short, src);
      snapshots.Tvmult(tmp_long, tmp_short);
      mass_matrix.Tvmult_add(dst, tmp_long);
    }

  private:
    SparseMatrix &mass_matrix;
    FullMatrix &snapshots;
  };

  class PODBasis
  {
  public:
    std::map<int, dealii::Vector<double>> vectors;
    std::vector<double> singular_values;
  };

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

    FullMatrix snapshot_matrix(snapshots.size(), mass_matrix.m());
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

    MethodOfSnapshots pod_data(petsc_mass_matrix, snapshot_matrix);
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
}