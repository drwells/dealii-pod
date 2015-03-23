plot-y-velocity
===========
Goals
-----
The goal of this application is to extract the second component of the velocity
for plotting purposes. Usually one examines the `y` velocity as a "poor man's"
vorticity.

Required Files
--------------
By default, this application assumes that `triangulation.txt` (the standard text
serialization of the triangulation) is in the current directory. It also assumes
(also by default)that the snapshots are in the working directory and match the
glob `snapshot-*h5`. Both the file name and the glob may be changed in the
configuration file.

Output
------
This application outputs HDF5 files containing the `y` component of the snapshot
files as well as the related `XDMF` file.
