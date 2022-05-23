# MTCNN Tensorflow C++ Implementation

This is a project to implement MTCNN, a perfect face detect algorithm, based on Tensorflow

This version is an exmaple of inside tensorflow code repository and use the bazel to build

The example outside the  will be developped  soon.


Here is to the trick to reuse pre-allocated buffer or tensor:

In order to create a tensor with pre-allocated buffer, a friend class of Tensor has to been defined.

Since TensorCApi is defined for C API usage, only used in libtensorflow.so, we can re-define this class in our file safely. 


# Build & Run

1 copy the mtcnn directory to tensorflow/exmaple

2 cd tensorflow/example/mtcnn

3 bazel build //tensorflow/example/mtcnn 

4 run: 

bazel-bin/tensorflow/examples/mtcnn/mtcnn  --image=photo_fname --graph=./tensorflow/examples/mtcnn/models/mtcnn_frozen_model.pb




# Credit

### MTCNN algorithm

https://github.com/kpzhang93/MTCNN_face_detection_alignment

### MTCNN C++ on Caffe

https://github.com/wowo200/MTCNN

### MTCNN python on Tensorflow 

FaceNet uses MTCNN to align face

https://github.com/davidsandberg/facenet
From this directory:
  facenet/src/align


