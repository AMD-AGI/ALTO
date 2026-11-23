#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR/../..

NGPU=1 \
MODULE=flux \
CONFIG=flux_debugmodel_lpt \
./examples/run.sh
