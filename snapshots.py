#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import print_function
import glob
import os
import sys
import numpy as np

def main():
    for solution in glob.glob("solution-*.vtk"):
        with open(solution, 'r') as file_handle:
            found_data = False
            for line in file_handle:
                if found_data:
                    break
                if 'LOOKUP_TABLE' in line:
                    found_data = True
                else:
                    pass

            if found_data:
                snapshot = np.fromstring(line, sep=' ')
                file_name = "snapshot-" + solution[9:-4] + ".txt"
                np.savetxt(file_name, snapshot)
            else:
                raise ValueError

if __name__ == '__main__':
    main()
