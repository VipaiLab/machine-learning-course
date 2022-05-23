#!/usr/bin/env sh
set -e

BUILD=/home2/huhaoji/caffe-master/build/tools
DATA=/home2/huhaoji/caffe-master/examples/mnistNew
$BUILD/caffe test -model $DATA/lenet_train_test.prototxt -weights $DATA/lenet_iter_10000.caffemodel -iterations 100 $@
