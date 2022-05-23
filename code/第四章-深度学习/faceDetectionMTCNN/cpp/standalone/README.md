# MTCNN Tensorflow C++ Implementation

This is a project to implement MTCNN, a perfect face detect algorithm, based on Tensorflow

This version is an exmaple of outside tensorflow code repository and call tensorflow service by libtensorflow.so

This is an example of how to tensorflow C API

# Build & Run

1 build tensorflow library distribution by following command in tensorflow directory
   
    bazel build --config=opt //tensorflow/tools/lib_package:libtensorflow

  the tarball, bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz, includes the libtensorflow.so and c header files 
  
2 edit Makefile, set TENSORFLOW_ROOT to the correct path in your machine

3  make
   two demoes will be created: "test" to check single photo while "camera" to do live face detection.

 
4 run: 
   arguments for test:

   test [ -i input_image ] [ -o output_image]  [ -m model_fname] [-s]

         -s     save the detected face into .jpg file  

   



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


