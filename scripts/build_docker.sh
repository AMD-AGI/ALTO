#!/bin/bash

IMAGE=wanghanthu/alto:rocm7.2.2-nightly-20260429
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..

docker build -t $IMAGE .
