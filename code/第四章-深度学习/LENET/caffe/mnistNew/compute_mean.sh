#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
set -e

DATA=/home2/huhaoji/caffe-master/examples/mnistNew
BUILD=/home2/huhaoji/caffe-master/build/tools

rm -rf $DATA/mean.binaryproto

$BUILD/compute_image_mean $DATA/mnist_train_lmdb $DATA/mean.binaryproto $@

