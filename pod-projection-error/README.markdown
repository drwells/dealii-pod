project
=======
Goals
-----
The goal of this application is to demonstrate the 'classic' POD interpolation
error formula 'the interpolation error equals the sum of the remaining
eigenvalues'.

Required Files
--------------
This application assumes that `triangulation.txt` (the standard text
serialization of the triangulation) and `mean-vector.h5` are in the current
directory. It also assumes that the POD vectors and snapshots are in the working
directory and match the globs `pod-vector-*h5` and `snapshot-*h5` respectively.

Output
------
This application prints out the projection errors for projection with one POD
vector, two POD vectors, etc. all the way up to the number of POD vectors in
the working directory.
