plot-pod-snapshots
==================
This program takes a 2D array, described by the file
`pod_coefficients_file_name`, and plots the corresponding ROM solution. Like the
other programs, the relevant POD vectors, mean vector, and triangulation must be
in the current directory for this to execute correctly.

If there are too many POD vectors in the current directory (i.e., more POD
vectors than columns in `pod_coefficients_file_name`), then the extra POD
vectors are ignored.
