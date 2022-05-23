/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// To use it, compile and then run in a working directory with the
// learning/brain/tutorials/label_image/data/ folder below it, and you should
// see the top five labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//
// The googlenet_graph.pb file included by default is created from Inception.
//
// Note that, for GIF inputs, to reuse existing code, only single-frame ones
// are supported.

#include <fstream>
#include <utility>
#include <vector>


#include "tensorflow/c/c_api.h"
#include "tensorflow_mtcnn.hpp"
#include "mtcnn.hpp"
#include "comm_lib.hpp"
#include "utils.hpp"

using std::string;

int main(int argc, char* argv[]) 
{
	string model_fname ="./models/mtcnn_frozen_model.pb";

	cv::VideoCapture camera;

	camera.open(0);

	if(!camera.isOpened())
	{
		std::cerr<<"failed to open camera"<<std::endl;
		return 1;
	}


	TF_Session * sess;
	TF_Graph * graph;


	sess=load_graph(model_fname.c_str(),&graph);

	if(sess==nullptr)
		return 1;


	cv::Mat frame;


	while(1)
	{

		camera.read(frame);

		std::vector<face_box> face_info;

		unsigned long start_time=get_cur_time();

		mtcnn_detect(sess,graph,frame,face_info);

		unsigned long end_time=get_cur_time();



		for(unsigned int i=0;i<face_info.size();i++)
		{
			face_box& box=face_info[i];

			/*draw box */

			cv::rectangle(frame, cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 1);


			/* draw landmark */

			for(int l=0;l<5;l++)
			{
				cv::circle(frame,cv::Point(box.landmark.x[l],box.landmark.y[l]),1,cv::Scalar(0, 0, 255),2);

			}
		}

		std::cout<<"total detected: "<<face_info.size()<<" faces. used "<<(end_time-start_time)<<" us"<<std::endl;

		cv::imshow("camera",frame);

		cv::waitKey(1000);
	}

	TF_Status* s = TF_NewStatus();

	TF_CloseSession(sess,s);
	TF_DeleteSession(sess,s);
	TF_DeleteGraph(graph);
	TF_DeleteStatus(s);

	return 0;
}







