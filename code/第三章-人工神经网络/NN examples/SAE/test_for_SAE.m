clear;clc;close all;
addpath(genpath('F:\project\M_DeepLearing'));
load mnist_uint8

train_x = double(train_x) / 255;

%create a sparse autoencoder
vsize = size(train_x,2);
hsize = 196;
sae = sae_create([vsize,hsize]);
sae.active_function = 'sigmoid';

option.batch_size = 100;
option.iteration = 400;
sae = sae_train(sae,option,train_x);
figure;
visualize(sae.W{1}');

