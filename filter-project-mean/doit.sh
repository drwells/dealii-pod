#!/bin/bash
set -euo pipefail

for FILTER_RADIUS in $(seq 0 100 | awk '{print $1/100}')
do
    perl -pi -e "s/(filter_radius = ).*/\1 $FILTER_RADIUS/" parameter-file.prm
    make release 1>/dev/null
    make run | tail -n +3 | head -n 13 >> log.txt
done
