clear;clc;close all;
%addpath(genpath('F:\mayanzhao\M_DeepLearing'));
load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);


%create a net
autoencoder = nn_create([784,400,169,49,10]);
%dnn = dnn_adjust(dnn,train_x,train_y);
%[wrongs,success_ratio,dnn] = nn_test(dnn,test_x,test_y);
% train
autoencoder = autoencoder_train(autoencoder,train_x,train_y);
%adjust
autoencoder = autoencoder_adjust(autoencoder,train_x,train_y);
%test

disp(['success rate is ',num2str(success_ratio)]);
figure;
visualize(autoencoder.W{1}');
figure;
visualize(autoencoder.W{2}');