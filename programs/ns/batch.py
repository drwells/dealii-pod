#!/usr/bin/env python
from __future__ import print_function

import math
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

FNULL = open(os.devnull, 'w')

class ParameterHandler(object):
    def __init__(self, parameter_file_name):
        self.parameter_file_name = parameter_file_name
        self.modified_parameters = dict()

    def add_modified_parameter(self, key, value):
        self.modified_parameters[key] = value

    def print_with_modifications(self, output_file_name):
        """Print the known parameter file to a file and modify any stored parameters.
        """
        with open(self.parameter_file_name, 'r') as input_handle:
            with open(output_file_name, 'w') as output_handle:
                for line in input_handle:
                    for key, value in self.modified_parameters.items():
                        if (line.find("set {}".format(key)) != -1):
                            # TODO make sure it is not a comment line
                            line = "  set {} = {}".format(key, value) + os.linesep
                    print(line, file=output_handle, end="")


def chunks(super_list, n):
    """Yield successive n-sized chunks from `super_list`."""
    for i in range(0, len(super_list), n):
        yield super_list[i:i+n]


# TODO this is not a particularly good approach. A much better way to do this
# would be to have exactly one object describing each parameter set and then put
# all of those objects in a queue.
def main():
    # noise_multipliers = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    noise_multipliers = [0.0, 1e-5]
    lavrentiev_parameters = np.linspace(0.01, 0.02, 101)
    filter_radii = np.linspace(0.05, 0.15, 101)
    n_processes = 16

    link_names = (["triangulation.txt", "mean-vector.h5", "initial.h5"] +
                  ["pod-vector-000000{}.h5".format(index) for index in range(6)])
    process_executable = "run.sh"
    copy_names = ["rom", process_executable]

    for noise_multiplier in noise_multipliers:
        for filter_radius in filter_radii:
            # split the innermost chunk over multiple processes
            for lavrentiev_chunk in chunks(lavrentiev_parameters, n_processes):
                processes = list()
                for lavrentiev_parameter in lavrentiev_chunk:
                    directory = tempfile.mkdtemp("-pod-ad-batch")
                    parameter_handler = ParameterHandler("parameters.prm")
                    parameter_handler.add_modified_parameter("noise_multiplier", noise_multiplier)
                    parameter_handler.add_modified_parameter("lavrentiev_parameter", lavrentiev_parameter)
                    parameter_handler.add_modified_parameter("filter_radius", filter_radius)
                    parameter_handler.print_with_modifications(directory + os.sep + "parameters.prm")
                    for link_name in link_names:
                        os.symlink(os.getcwd() + os.sep + link_name, directory + os.sep + link_name)
                    for copy_name in copy_names:
                        shutil.copy2(copy_name, directory + os.sep + copy_name)
                    # executable does all the work from here: compiles, runs, and moves
                    # the output to the correct directory
                    processes.append(
                        [subprocess.Popen(directory + os.sep + process_executable, stdout=FNULL),
                        directory])
                    print("set up noise = {}, radius = {}, and lavrentiev = {}"
                          .format(noise_multiplier, filter_radius, lavrentiev_parameter))

                while 0 < len(processes):
                    for index in range(len(processes)):
                        process, directory = processes[index]
                        if process.poll() is not None:
                            processes.pop(index)
                            shutil.rmtree(directory)
                            break
                        else:
                            time.sleep(1.0)
                            continue

if __name__ == '__main__':
    main()
