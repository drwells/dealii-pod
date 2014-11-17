#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import glob
import numpy as np

def main():
    for solution in glob.glob("solution-*.vtk"):
        with open(solution, 'r') as vtk_file_handle:
            save_line = False
            num_points = 0
            output_file_name = "snapshot-" + solution[9:-4] + ".txt"
            output_file_handle = None

            for line in vtk_file_handle:
                if save_line:
                    if "SCALARS" in line:
                        break
                    else:
                        output_file_handle.write(line)
                else:
                    if 'POINTS' in line:
                        if num_points == 0:
                            num_points = int(line.split(' ')[1])
                        else:
                            new_num_points = int(line.split(' ')[1])
                            if num_points != new_num_points:
                                raise ValueError(
                                    "The number of mesh was different in the "
                                    "mesh ({}) and data ({}) sections."
                                    .format(num_points, new_num_points))
                    if 'VECTORS' in line:
                        save_line = True
                        output_file_handle = open(output_file_name, 'w')

        output_file_handle.close()
        # for ROM purposes it is useful to have the whole snapshot as one big vector.
        concatenated_file_name = "snapshot-concatenated-" + solution[9:-4] + ".txt"
        snapshot = np.loadtxt(output_file_name)
        np.savetxt(concatenated_file_name, snapshot.flatten('F'))

if __name__ == '__main__':
    main()
