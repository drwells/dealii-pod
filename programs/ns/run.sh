#!/bin/bash

# ensure that we are in the same directory as the current script.
SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SOURCE_DIR
./ns-rom

mv pod-ad-*.h5 ~/Data/PODAD/
