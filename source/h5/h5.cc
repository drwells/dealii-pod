#include <deal.II-pod/h5/h5.templates.h>

namespace POD
{
  using namespace dealii;
  namespace H5
  {
    template
    void load_block_vector(const std::string &file_name,
                           BlockVector<double> &block_vector);

    template
    void save_block_vector(const std::string &file_name,
                           BlockVector<double> &block_vector);

    template
    void load_full_matrix(const std::string &file_name,
                          FullMatrix<double> &matrix);

    template
    void load_full_matrix(const std::string &file_name,
                          LAPACKFullMatrix<double> &matrix);

    template
    void save_full_matrix(const std::string &file_name,
                          const FullMatrix<double> &matrix);

    template
    void save_full_matrix(const std::string &file_name,
                          const LAPACKFullMatrix<double> &matrix);

    template
    void load_vector(const std::string &file_name,
                     Vector<double> &vector);

    template
    void save_vector(const std::string &file_name,
                     const Vector<double> &vector);

    template
    void load_full_matrices(const std::string &file_name,
                            std::vector<FullMatrix<double>> &matrices);

    template
    void save_full_matrices(const std::string &file_name,
                            const std::vector<FullMatrix<double>> &matrices);
  }
}
