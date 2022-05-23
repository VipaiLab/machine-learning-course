clear;clc;close all;
%addpath(genpath('D:\matlab\NN example autoencoder\data'));
load mnist_uint8;                %数据
train_x = double(train_x) / 255; %归一化
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);


% normalize


%create a netual net
nn = nn_create([784,40,10]);%创建最简单的3层网络，出入层784，隐层40，输出10
%train
option.batch_size = 100;%一批取100组数据
option.iteration = 5;%迭代5次
nn = nn_train(nn,option,train_x,train_y);
%test
[wrongs,ratio] = nn_test(nn,test_x,test_y);
disp([num2str(size(test_x,1)) ' photos have been tested, the success ratio is ' num2str(ratio)]);
