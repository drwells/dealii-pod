rom-error
=========
Goals
-----
The goal of this application is to compute the L2 error in the ROM solution.

Required Files
--------------
All of these settings may be changed by editing `parameters.prm`. By
default, this application assumes that `triangulation.txt` (the standard text
serialization of the triangulation) is in the current directory. It also assumes
that the snapshots are in the working directory and match the glob
`snapshot-*h5`. These snapshots describe the DNS solution.

In addition, the ROM solution is present in an array of coefficients stored in
`rom-solution.h5`. The POD basis is stored in `pod-vectors*h5` and
`mean-vector.h5`.

Required Configuration
----------------------
Since both the ROM solution (stored as rows in a HDF5 matrix) and the snapshots
(organized by integer index) are compared, the configuration requires
`start_time` and `stop_time` be set for both the ROM and the DNS solutions. The
solver assumes that the data is linearly spaced and will correctly sample the
more dense (usually the ROM) data set.

Output
------
This application prints the L2 error to the screen.
