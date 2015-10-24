ns
==
Goals
-----
This is the Navier Stokes ROM and various supporting files. This application
runs the ROM and saves the POD coefficients to an HDF5 file.

Included Files
--------------
This program relies a lot on supporting functions. The ones directly related to
either the ROM or Navier Stokes reside in this folder:
* `filter.h`: implementation of the Leray finite element filter
* `ns.h`: various functions for Navier Stokes
* `parameters.h`: standard parameters file
* `rk_factory.h`: factory function for setting up the various ROMs
where each header file has a corresponding source file.

Required Files
--------------
This application assumes that `triangulation.txt` (the standard text
serialization of the triangulation) is in the current directory. It also assumes
the POD vectors are in the working directory and match the glob
`pod-vectors-*h5`. Finally, it assumes that the mean vector is available at
`mean-vector.h5` in the current working directory.

Output
------
This application saves the POD coefficients in a file whose name depends on the
solver configuration. See the source of `rk_factory.cc` for more details.
