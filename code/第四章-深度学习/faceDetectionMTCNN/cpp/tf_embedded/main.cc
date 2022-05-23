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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "mtcnn.hpp"
#include "comm_lib.hpp"
#include "utils.hpp"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using tensorflow::TensorBuffer;
using tensorflow::DT_FLOAT;
using tensorflow::TensorShape;
using tensorflow::AllocationDescription;


// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
		std::unique_ptr<tensorflow::Session>* session) {
	tensorflow::GraphDef graph_def;
	Status load_graph_status =
		ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
	if (!load_graph_status.ok()) {
		return tensorflow::errors::NotFound("Failed to load compute graph at '",
				graph_file_name, "'");
	}
	session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
	Status session_create_status = (*session)->Create(graph_def);
	if (!session_create_status.ok()) {
		return session_create_status;
	}
	return Status::OK();
}


/* copied from TF_ManagedBuffer in c_api.cc */
class my_buffer: public TensorBuffer{

	public:
		float * data_;
		size_t  len_;

		~my_buffer() override {
			//nothing to do
		}

		void* data() const override { return data_; }
		size_t size() const override { return len_; }
		bool OwnsMemory() const override { return false; }

		TensorBuffer* root_buffer() override { return this; }

		void FillAllocationDescription(AllocationDescription* proto) const override {
			tensorflow::int64 rb = size();
			proto->set_requested_bytes(rb);
			proto->set_allocator_name(tensorflow::cpu_allocator()->Name());
		}

};


class tensorflow::TensorCApi {
	public:
		static Tensor create_tensor(DataType dtype, const TensorShape& shape, TensorBuffer* buf)
		{
			return  Tensor(dtype,shape,buf);
		}

};




void generate_bounding_box_tf(const float * confidence_data, int confidence_size,
		const float * reg_data, float scale, float threshold, 
		int feature_h, int feature_w, std::vector<face_box>&  output, bool transposed)
{

	int stride = 2;
	int cellSize = 12;

	int img_h= feature_h;
	int img_w = feature_w;


	for(int y=0;y<img_h;y++)
		for(int x=0;x<img_w;x++)
		{
			int line_size=img_w*2;

			float score=confidence_data[line_size*y+2*x+1];

			if(score>= threshold)
			{

				float top_x = (int)((x*stride + 1) / scale);
				float top_y = (int)((y*stride + 1) / scale);
				float bottom_x = (int)((x*stride + cellSize) / scale);
				float bottom_y = (int)((y*stride + cellSize) / scale);

				face_box box;

				box.x0 = top_x;
				box.y0 = top_y;
				box.x1 = bottom_x;
				box.y1 = bottom_y;

				box.score=score;

				int c_offset=(img_w*4)*y+4*x;

				if(transposed)
				{

					box.regress[1]=reg_data[c_offset];
					box.regress[0]=reg_data[c_offset+1]; 
					box.regress[3]=reg_data[c_offset+2];
					box.regress[2]= reg_data[c_offset+3];
				}
				else {

					box.regress[0]=reg_data[c_offset];
					box.regress[1]=reg_data[c_offset+1]; 
					box.regress[2]=reg_data[c_offset+2];
					box.regress[3]= reg_data[c_offset+3];
				}

				output.push_back(box);
			}

		}
}



void run_PNet(std::unique_ptr<tensorflow::Session>& sess, cv::Mat& img, scale_window& win, std::vector<face_box>& box_list)
{
	cv::Mat  resized;
	int scale_h=win.h;
	int scale_w=win.w;
	float scale=win.scale;
	float pnet_threshold=0.6;


	cv::resize(img, resized, cv::Size(scale_w, scale_h),0,0);

	/* tensorflow related*/

	const int64_t dim[4] = {1,scale_h,scale_w,3};

	my_buffer tensor_buf;


	tensor_buf.data_=(float *)resized.ptr();
	tensor_buf.len_=scale_h*scale_w*3;

	std::vector<tensorflow::int64> tensor_dim;

	for(int i=0;i<4;i++)
		tensor_dim.push_back(dim[i]);


	Tensor input_tensor=tensorflow::TensorCApi::create_tensor(DT_FLOAT,TensorShape(tensor_dim), &tensor_buf);


	std::vector<Tensor> output_tensor;


	std::vector<std::pair<string, Tensor> > input_tname;

	std::pair<string,Tensor> input0("pnet/input:0",input_tensor);
	input_tname.push_back(input0);

	std::vector<string> output_tname;

	output_tname.push_back("pnet/conv4-2/BiasAdd:0");
	output_tname.push_back("pnet/prob1:0");

	std::vector<string> output_node;



	Status run_status = sess->Run(input_tname,output_tname,output_node,&output_tensor);

	if(!run_status.ok())
	{
		std::cerr<<"run PNet error"<<std::endl;
		return;
	}

	/*retrieval the forward results*/

	TensorShape reg_shape=output_tensor[0].shape();
	TensorShape conf_shape=output_tensor[1].shape();

	int feature_h=reg_shape.dim_size(1);
	int feature_w=reg_shape.dim_size(2);


	std::vector<face_box> candidate_boxes;

	const tensorflow::StringPiece conf_piece=output_tensor[1].tensor_data();
	const tensorflow::StringPiece reg_piece=output_tensor[0].tensor_data();


	const float * conf_data=(const float *)conf_piece.data();
	int conf_size=feature_h*feature_w*2;
	const float * reg_data=(const float *)reg_piece.data();

	generate_bounding_box_tf(conf_data,conf_size,reg_data, 
			scale,pnet_threshold,feature_h,feature_w,candidate_boxes,true);


	nms_boxes(candidate_boxes, 0.5, NMS_UNION,box_list);

}



void copy_one_patch(const cv::Mat& img,face_box&input_box,float * data_to, int height, int width)
{
	cv::Mat resized(height,width,CV_32FC3,data_to);


	cv::Mat chop_img = img(cv::Range(input_box.py0,input_box.py1),
			cv::Range(input_box.px0, input_box.px1));

	int pad_top = std::abs(input_box.py0 - input_box.y0);
	int pad_left = std::abs(input_box.px0 - input_box.x0);
	int pad_bottom = std::abs(input_box.py1 - input_box.y1);
	int pad_right = std::abs(input_box.px1-input_box.x1);

	cv::copyMakeBorder(chop_img, chop_img, pad_top, pad_bottom,pad_left, pad_right,  cv::BORDER_CONSTANT, cv::Scalar(0));

	cv::resize(chop_img,resized, cv::Size(width, height), 0, 0);
}


void run_RNet(std::unique_ptr<tensorflow::Session>& sess, cv::Mat& img, std::vector<face_box>& pnet_boxes, std::vector<face_box>& output_boxes)
{
	int batch=pnet_boxes.size();
	int channel = 3;
	int height = 24;
	int width = 24;

	float rnet_threshold=0.7;

	/* prepare input image data */

	int  input_size=batch*height*width*channel;

	std::vector<float> input_buffer(input_size);

	float * input_data=input_buffer.data();

	for(int i=0;i<batch;i++)
	{
		int patch_size=width*height*channel;

		copy_one_patch(img,pnet_boxes[i], input_data,height,width);

		input_data+=patch_size;
	}


	/* tensorflow  related */
	const int64_t dim[4] = {batch,height,width,channel};


	my_buffer tensor_buf;

	tensor_buf.data_=input_buffer.data();
	tensor_buf.len_=input_size;


	std::vector<tensorflow::int64> tensor_dim;

	for(int i=0;i<4;i++)
		tensor_dim.push_back(dim[i]);

	Tensor input_tensor=tensorflow::TensorCApi::create_tensor(DT_FLOAT,TensorShape(tensor_dim), &tensor_buf);

	std::vector<Tensor> output_tensor;


	std::vector<std::pair<string, Tensor> > input_tname;

	std::pair<string,Tensor> input0("rnet/input:0",input_tensor);
	input_tname.push_back(input0);

	std::vector<string> output_tname;

	output_tname.push_back("rnet/conv5-2/conv5-2:0");
	output_tname.push_back("rnet/prob1:0");

	std::vector<string> output_node;

	Status run_status = sess->Run(input_tname,output_tname,output_node,&output_tensor);

	if(!run_status.ok())
	{
		std::cerr<<"run PNet error"<<std::endl;
		return;
	}

	/*retrieval the forward results*/

	const tensorflow::StringPiece conf_piece=output_tensor[1].tensor_data();
	const tensorflow::StringPiece reg_piece=output_tensor[0].tensor_data();

	const float * conf_data=(const float *)conf_piece.data();
	const float * reg_data=(const float *)reg_piece.data();

	for(int i=0;i<batch;i++)
	{

		if(conf_data[1]>rnet_threshold)
		{
			face_box output_box;

			face_box& input_box=pnet_boxes[i];

			output_box.x0=input_box.x0;
			output_box.y0=input_box.y0;
			output_box.x1=input_box.x1;
			output_box.y1=input_box.y1;

			output_box.score = *(conf_data+1);

			/*Note: regress's value is swaped here!!!*/

			output_box.regress[0]=reg_data[1];
			output_box.regress[1]=reg_data[0];
			output_box.regress[2]=reg_data[3];
			output_box.regress[3]=reg_data[2];

			output_boxes.push_back(output_box);


		}

		conf_data+=2;
		reg_data+=4;

	}

}

void run_ONet(std::unique_ptr<tensorflow::Session>& sess, cv::Mat& img, std::vector<face_box>& rnet_boxes, std::vector<face_box>& output_boxes)
{
	int batch=rnet_boxes.size();
	int channel = 3;
	int height = 48;
	int width = 48;

	float onet_threshold=0.9;

	/* prepare input image data */

	int  input_size=batch*height*width*channel;

	std::vector<float> input_buffer(input_size);

	float * input_data=input_buffer.data();

	for(int i=0;i<batch;i++)
	{
		int patch_size=width*height*channel;

		copy_one_patch(img,rnet_boxes[i], input_data,height,width);

		input_data+=patch_size;
	}


	/* tensorflow  related */
	const int64_t dim[4] = {batch,height,width,channel};


	my_buffer tensor_buf;

	tensor_buf.data_=input_buffer.data();
	tensor_buf.len_=input_size;


	std::vector<tensorflow::int64> tensor_dim;

	for(int i=0;i<4;i++)
		tensor_dim.push_back(dim[i]);

	Tensor input_tensor=tensorflow::TensorCApi::create_tensor(DT_FLOAT,TensorShape(tensor_dim), &tensor_buf);

	std::vector<Tensor> output_tensor;


	std::vector<std::pair<string, Tensor> > input_tname;

	std::pair<string,Tensor> input0("onet/input:0",input_tensor);
	input_tname.push_back(input0);

	std::vector<string> output_tname;

	output_tname.push_back("onet/conv6-2/conv6-2:0");
	output_tname.push_back("onet/conv6-3/conv6-3:0");
	output_tname.push_back("onet/prob1:0");

	std::vector<string> output_node;

	Status run_status = sess->Run(input_tname,output_tname,output_node,&output_tensor);

	if(!run_status.ok())
	{
		std::cerr<<"run PNet error"<<std::endl;
		return;
	}

	/*retrieval the forward results*/

	const tensorflow::StringPiece conf_piece=output_tensor[2].tensor_data();
	const tensorflow::StringPiece reg_piece=output_tensor[0].tensor_data();
	const tensorflow::StringPiece points_piece=output_tensor[1].tensor_data();

	const float * conf_data=(const float *)conf_piece.data();
	const float * reg_data=(const float *)reg_piece.data();
	const float * points_data=(const float *)points_piece.data();

	for(int i=0;i<batch;i++)
	{

		if(conf_data[1]>onet_threshold)
		{
			face_box output_box;

			face_box& input_box=rnet_boxes[i];

			output_box.x0=input_box.x0;
			output_box.y0=input_box.y0;
			output_box.x1=input_box.x1;
			output_box.y1=input_box.y1;

			output_box.score = conf_data[1];

			output_box.regress[0]=reg_data[1];
			output_box.regress[1]=reg_data[0];
			output_box.regress[2]=reg_data[3];
			output_box.regress[3]=reg_data[2];

			/*Note: switched x,y points value too..*/
			for (int j = 0; j<5; j++){
				output_box.landmark.x[j] = *(points_data + j+5);
				output_box.landmark.y[j] = *(points_data + j);
			}

			output_boxes.push_back(output_box);


		}

		conf_data+=2;
		reg_data+=4;
		points_data+=10;
	}

}

void mtcnn_detect(std::unique_ptr<tensorflow::Session>& sess, cv::Mat& img, std::vector<face_box>& face_list)
{
	cv::Mat working_img;

	float alpha=0.0078125;
	float mean=127.5;



	img.convertTo(working_img, CV_32FC3);

	working_img=(working_img-mean)*alpha;

	working_img=working_img.t();

	cv::cvtColor(working_img,working_img, cv::COLOR_BGR2RGB);

	int img_h=working_img.rows;
	int img_w=working_img.cols;


	int min_size=40;
	float factor=0.709;


	std::vector<scale_window> win_list;

	std::vector<face_box> total_pnet_boxes;
	std::vector<face_box> total_rnet_boxes;
	std::vector<face_box> total_onet_boxes;


	cal_pyramid_list(img_h,img_w,min_size,factor,win_list);

	for(unsigned int i=0;i<win_list.size();i++)
	{
		std::vector<face_box>boxes;

		run_PNet(sess,working_img,win_list[i],boxes);

		total_pnet_boxes.insert(total_pnet_boxes.end(),boxes.begin(),boxes.end());
	}


	std::vector<face_box> pnet_boxes;
	process_boxes(total_pnet_boxes,img_h,img_w,pnet_boxes);


	// RNet
	std::vector<face_box> rnet_boxes;

	run_RNet(sess,working_img, pnet_boxes,total_rnet_boxes);

	process_boxes(total_rnet_boxes,img_h,img_w,rnet_boxes);


	//ONet
	run_ONet(sess,working_img, rnet_boxes,total_onet_boxes);

	//calculate the landmark

	for(unsigned int i=0;i<total_onet_boxes.size();i++)
	{
		face_box& box=total_onet_boxes[i];

		float h=box.x1-box.x0+1;
		float w=box.y1-box.y0+1;

		for(int j=0;j<5;j++)
		{
			box.landmark.x[j]=box.x0+w*box.landmark.x[j]-1;
			box.landmark.y[j]=box.y0+h*box.landmark.y[j]-1;
		}

	}


	//Get Final Result
	regress_boxes(total_onet_boxes);
	nms_boxes(total_onet_boxes, 0.7, NMS_MIN,face_list);

	//switch x and y, since working_img is transposed

	for(unsigned int i=0;i<face_list.size();i++)
	{
		face_box& box=face_list[i];

		std::swap(box.x0,box.y0);
		std::swap(box.x1,box.y1);

		for(int l=0;l<5;l++)
		{
			std::swap(box.landmark.x[l],box.landmark.y[l]);
		}
	}


}



int main(int argc, char* argv[]) {
	string image = "./test.jpg";
	string graph =
		"./models/mtcnn_frozen_model.pb";

        string output_fname="./new.jpg";

	string root_dir = "";
	std::vector<Flag> flag_list = {
		Flag("image", &image, "image to be processed"),
		Flag("graph", &graph, "graph to be executed"),
		Flag("output", &graph, "image with face boxed"),
	};

	string usage = tensorflow::Flags::Usage(argv[0], flag_list);
	const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
	if (!parse_result) {
		LOG(ERROR) << usage;
		return -1;
	}

	// We need to call this to set up global state for TensorFlow.
	tensorflow::port::InitMain(argv[0], &argc, &argv);
	if (argc > 1) {
		LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
		return -1;
	}

	// First we load and initialize the model.
	std::unique_ptr<tensorflow::Session> session;
	string graph_path = tensorflow::io::JoinPath(root_dir, graph);
	Status load_graph_status = LoadGraph(graph_path, &session);
	if (!load_graph_status.ok()) {
		LOG(ERROR) << load_graph_status;
		return -1;
	}

	//Load image

	cv::Mat frame = cv::imread(image);

	if(!frame.data)
	{
		std::cerr<<"failed to read image file: "<<image<<std::endl;
		return 1;
	}


	std::vector<face_box> face_info;

	unsigned long start_time=get_cur_time();

	mtcnn_detect(session,frame,face_info);

	unsigned long end_time=get_cur_time();

	int save_chop=0;


	for(unsigned int i=0;i<face_info.size();i++)
	{
		face_box& box=face_info[i];

		printf("face %d: x0,y0 %2.5f %2.5f  x1,y1 %2.5f  %2.5f conf: %2.5f\n",i,
				box.x0,box.y0,box.x1,box.y1, box.score);
		printf("landmark: ");

		for(unsigned int j=0;j<5;j++)
			printf(" (%2.5f %2.5f)",box.landmark.x[j], box.landmark.y[j]);

		printf("\n");


		if(save_chop)
		{

			cv::Mat corp_img=frame(cv::Range(box.y0,box.y1),
					cv::Range(box.x0,box.x1));

			char title[128];

			sprintf(title,"id%d.jpg",i);

			cv::imwrite(title,corp_img);
		}

		/*draw box */

		cv::rectangle(frame, cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 1);


		/* draw landmark */

		for(int l=0;l<5;l++)
		{
			cv::circle(frame,cv::Point(box.landmark.x[l],box.landmark.y[l]),1,cv::Scalar(0, 0, 255),2);

		}
	}

	cv::imwrite(output_fname,frame);

	std::cout<<"total detected: "<<face_info.size()<<" faces. used "<<(end_time-start_time)<<" us"<<std::endl;
	std::cout<<"boxed faces are in file: "<<output_fname<<std::endl;

	return 0;
}







