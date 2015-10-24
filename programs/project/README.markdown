project
=======
Goals
-----
The goal of this application is to calculate the $L^2$ norms of the fluctuations
and the projected POD coefficients of the snapshots.

Required Files
--------------
This application assumes that `triangulation.txt` (the standard text
serialization of the triangulation) and `mean-vector.h5` are in the current
directory. It also assumes that the POD vectors and snapshots are in the working
directory and match the globs `pod-vector-*h5` and `snapshot-*h5` respectively.

Output
------
This application saves the fluctuations at each snapshot to
`fluctuation-norms.h5` and the coefficients resulting from projecting the
snapshots onto the POD vectors to `projected-pod-coefficients.h5`.