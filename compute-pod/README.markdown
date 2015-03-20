compute-pod
===========
Goals
-----
The goal of this application is to compute the POD basis for a (possibly large)
set of snapshots.

Required Files
--------------
This application assumes that `triangulation.txt` (the standard text
serialization of the triangulation) is in the current directory. It also assumes
that the snapshots are in the working directory and match the glob
`snapshot-*h5`.

Output
------
This application outputs the POD vectors and mean vector calculated from the
given snapshots. It also saves an `XDMF` file and enough information to plot the
POD vectors.