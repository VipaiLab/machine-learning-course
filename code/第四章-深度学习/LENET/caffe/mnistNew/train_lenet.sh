#!/usr/bin/env sh
set -e

BUILD=/home2/huhaoji/caffe-master/build/tools
DATA=/home2/huhaoji/caffe-master/examples/mnistNew
$BUILD/caffe train --solver=$DATA/lenet_solver.prototxt $@
