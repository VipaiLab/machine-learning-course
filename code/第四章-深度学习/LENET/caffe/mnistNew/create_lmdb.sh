#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
set -e

DATA=/home2/huhaoji/caffe-master/examples/mnistNew
BUILD=/home2/huhaoji/caffe-master/build/tools

rm -rf $DATA/mnist_train_lmdb
rm -rf $DATA/mnist_test_lmdb

$BUILD/convert_imageset --shuffle \
--resize_height=28 --resize_width=28 \
$DATA/    \
$DATA/training.txt  $DATA/mnist_train_lmdb

$BUILD/convert_imageset --shuffle \
--resize_height=28 --resize_width=28 \
$DATA/    \
$DATA/testing.txt  $DATA/mnist_test_lmdb

