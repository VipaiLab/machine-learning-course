clear;clc;close all;
%addpath(genpath('F:\mayanzhao\M_DeepLearing'));
load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);
nn = nn_create([784,400,169,49,10],'active function','sigmoid','learning rate',0.01, 'batch normalization',1,'optimization method','Adam', 'objective function', 'Cross Entropy');

option.batch_size = 100;
option.iteration = 1;


iteration = 0;
maxAccuracy = 0;
totalAccuracy = [];
while(1)
    iteration = iteration +1; 
    nn = nn_train(nn,option,train_x,train_y);
    totalCost(iteration) = sum(nn.cost)/length(nn.cost);
   % plot(totalCost);
    [wrongs,accuracy] = nn_test(nn,test_x,test_y);
    totalAccuracy = [totalAccuracy,accuracy];
    if accuracy>maxAccuracy
        maxAccuracy = accuracy;
        storedNN = nn;
    end;
    cost = totalCost(iteration);
    accuracy
    cost
end;
[wrongs,accuracy] = nn_test(storedNN,train_x,train_y);
