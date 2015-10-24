filter-project-mean
===================
Goals
-----
The goal of this application is to examine the accuracy of representing the
*filtered* mean vector as a linear combination of POD vectors. Spoiler alert:
this does not work.

Required Files
--------------
By default, this application assumes that `triangulation.txt` (the standard text
serialization of the triangulation) is in the current directory. It also assumes
(also by default) that the POD vectors are in the working directory and match
the glob `pod-vectors-*h5`. Finally, it assumes that the mean vector is
available at `mean-vector.h5` in the current working directory. Both the file
names and the glob may be changed in the configuration file.

Output
------
This program saves the two (the FE and POD filter) representations of the
filtered mean vector. It also prints the POD filtered mean L2 norm, the FE
filtered mean L2 norm, and the L2 norm of the difference.
