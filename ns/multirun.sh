#!/usr/bin/env bash

POD_DIRECTORY_ROOT=/home/drwells/Documents/Code/CPP/dealii-rom/compute-pod/3d-data-re-100/

function copy_files()
{
    rm -f pod-vector-00*h5 initial.h5 mean-vector.h5 triangulation.txt
    ln -s $(ls -S $POD_DIRECTORY_ROOT/pod-vector-0*h5 | head -n $N_POD_VECTORS) ./
    ln -s $(ls -S $POD_DIRECTORY_ROOT/{initial,mean-vector}.h5) ./
    ln -s $(ls -S $POD_DIRECTORY_ROOT/triangulation.txt) ./
    printf 'copied POD vectors: %s\n' $(ls pod-vector-0*h5 | wc -l)
}

function run_filter_radii()
{
    local N_POD_VECTORS=$1
    local LINE_LEFT_PART='set filter_radius ='
    copy_files

    perl -pi -e "s/(set filter_model =).*/\1 Differential/" parameter-file.prm
    for FILTER_VALUE in $(seq 0 100 | awk '{print $1/100}')
    do
        perl -pi -e "s/($LINE_LEFT_PART) .*/\1 $FILTER_VALUE/" parameter-file.prm
        make release 1>/dev/null 2>/dev/null && make run 1>/dev/null
        printf "finished filter radius %s\n" $FILTER_VALUE
    done
}

function run_filter_cutoff_n()
{
    local N_POD_VECTORS=$1
    local LINE_LEFT_PART='set cutoff_n ='
    copy_files

    perl -pi -e "s/(set filter_model =).*/\1 L2Projection/" parameter-file.prm
    for CUTOFF_N in $(seq 0 $N_POD_VECTORS)
    do
        perl -pi -e "s/($LINE_LEFT_PART) .*/\1 $CUTOFF_N/" parameter-file.prm
        make release 1>/dev/null 2>/dev/null && make run 1>/dev/null
        printf "finished cutoff number %s\n" $CUTOFF_N
    done
}

function run_post_filter()
{
    local N_POD_VECTORS=$1
    local LINE_LEFT_PART='set filter_radius ='

    perl -pi -e "s/(set filter_model =).*/\1 PostFilter/" parameter-file.prm

    copy_files
    for FILTER_VALUE in $(seq 0 100 | awk '{print $1/100}')
    do
        perl -pi -e "s/($LINE_LEFT_PART) .*/\1 $FILTER_VALUE/" parameter-file.prm
        make release 1>/dev/null 2>/dev/null && make run
        printf "finished filter radius %s\n" $FILTER_VALUE
    done
}

ARGV=("$@")
ARGC=("$#")

MATCH_DIGITS="^[0-9]+$"
if ! [[ "${ARGV[1]}" =~ $MATCH_DIGITS ]]; then
    printf "The second argument (%s) must be a positive number.\n" ${ARGV[1]} >&2
    exit 1
fi

if [[ "${ARGV[1]}" -lt 1 ]]; then
    printf "The second argument (%s) must be a positive number.\n" ${ARGV[1]} >&2
    exit 1
fi


if [[ "${ARGV[0]}" == "prefilter" ]]; then
    run_filter_radii ${ARGV[1]}
    exit 0
elif [[ "${ARGV[0]}" == "cutoff" ]]; then
    run_filter_cutoff_n ${ARGV[1]}
    exit 0
elif [[ "${ARGV[0]}" == "postfilter" ]]; then
    run_post_filter ${ARGV[1]}
    exit 0
else
    printf "unrecognized filtering strategy (%s)\n" ${ARGV[0]} >&2
    exit 1
fi
