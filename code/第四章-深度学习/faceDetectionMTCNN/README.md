# tensorflow-mtcnn

MTCNN is one of the best face detection algorithms.
Here is inference only for MTCNN face detector on Tensorflow, which is based on davidsandberg's facenet project, include the python version and C++ version.

## C++

There are two version for C++.

One is to be build inside tensorflow code repository, so that it needs to be copied to the directory tensorflow/example.
please check cpp/tf_embedded/README.md for details.

The other is the standalone one, just needs libtensorflow.so and c_api.h to build and run.
Please check cpp/standalone/README.md for more details

## Python Run
1. install tensorflow first, please refers to https://www.tensorflow.org/install
2. install python packages: opencv, numpy
3. python ./facedetect_mtcnn.py --input input.jpg --output  new.jpg

## Build tensorflow on arm64 board

Please check out the guide [how to build tensorflow on firefly](https://cyberfire.github.io/tensorflow/rk3399/howto%20build%20tensorflow%20on%20firefly.md)

## Credit

### MTCNN algorithm

https://github.com/kpzhang93/MTCNN_face_detection_alignment

### MTCNN C++ on Caffe

https://github.com/wowo200/MTCNN

### MTCNN python on Tensorflow 

FaceNet uses MTCNN to align face

https://github.com/davidsandberg/facenet
From this directory:
  facenet/src/align


