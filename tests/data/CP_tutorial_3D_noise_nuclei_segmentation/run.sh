#!/bin/bash
set -euo pipefail

# build and run cellprofiler from a docker container
CPDOCKER_RUNDIR=$PWD/tests/data/CP_tutorial_3D_noise_nuclei_segmentation
CPDOCKER_IMAGE_NAME=cp-3d-nuclei-profiling

# Ensure dedicated output folder for segmentation masks exists.
mkdir -p "$CPDOCKER_RUNDIR/output/masks"

# build image
docker build --platform linux/amd64 -t "$CPDOCKER_IMAGE_NAME" -f "$CPDOCKER_RUNDIR/Dockerfile" .

# show the CellProfiler version and use run as a quick test
echo "CellProfiler version:"
docker run --rm --platform linux/amd64 -w /app \
    -v "$CPDOCKER_RUNDIR:/app" \
    "$CPDOCKER_IMAGE_NAME" \
    --version

# run cellprofiler with the examplehuman dataset from:
# https://cellprofiler.org/examples
docker run --rm --platform linux/amd64 -w /app \
    -v "$CPDOCKER_RUNDIR:/app" \
    "$CPDOCKER_IMAGE_NAME" \
    -c -r -p 3d-nuclei-profiling.cppipe -o output -i input
