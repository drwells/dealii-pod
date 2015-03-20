deal.II-POD
===========
This repository implements POD-ROMs (proper orthogonal decomposition (based)
reduced order models) with the `deal.II` finite element library. A README file
located in each directory contains further explanations about each subproject:
most of them are just numerical experiments.

This library consists of most of the (programming) work that I did to finish my
PhD thesis.

General Design
==============
Since I used this library in conjunction with a 3D Navier Stokes solver, I
anticipate that all snapshots, POD vectors, and the mean vector are calculated
ahead of time. Therefore nearly every component examines the working directory
with the `glob` function to load the relevant files. For example, the program
`compute-pod` locates snapshots by checking for matches to the glob
`snapshot-*h5` in the local directory.

All parameters for running each subproject are (or should be if they are not)
located in parameter files.

I make very extensive use of C++11 through out the entire project. I would be
hard-pressed to find a single function that does not use some feature of C++11.
If you do not believe that `unique_ptr` is the greatest thing ever (or you do
not have a compliant compiler) then this library is probably not for you.

Directory Contents
==================
compute-pod
-----------
This program computes the POD basis by the method of snapshots.
