# Description:
#   TensorFlow C++ inference example for labeling images.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_binary(
    name = "mtcnn",
    srcs = [
        "main.cc",
        "comm_lib.cpp",
        "comm_lib.hpp",
        "mtcnn.hpp",
        "utils.cpp",
        "utils.hpp"
    ],

    copts = [
         "-I/usr/local/include/opencv",
         " -I/usr/local/include",
         "-Wno-sign-compare",
         "-Wall",
         "-g",
         "-O0",
          ],

    linkopts = 
        ["-lm",
         "-L/usr/local/lib -lopencv_shape -lopencv_stitching -lopencv_objdetect -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_core",
         "-g",
        ],
    deps = 
        [
            "//tensorflow/cc:cc_ops",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:tensorflow",
        ],

)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "bin/**",
            "gen/**",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
