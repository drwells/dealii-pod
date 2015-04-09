#include "parameters.h"

Parameters::Parameters()
  {
    snapshot_glob = "snapshot-*h5";
    pod_vector_glob = "pod-vector-*h5";
    mean_vector_file_name = "mean-vector.h5";
    pod_coefficients_file_name = "test.h5";
    renumber = false;
    fe_order = 2;

    snapshot_start_time = 0.0;
    snapshot_stop_time = 500.0;

    rom_start_time = 30.0;
    rom_stop_time = 500.0;
  }
