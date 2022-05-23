clear;clc;close all;
load mnist_uint8;                %数据
train_x = double(train_x) / 255; %归一化
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);


% normalize


%create a netual net
nn = nn_create([784,400,169,49,10],'active function','sigmoid','learning rate',0.01, 'batch normalization',1,'optimization method','Adam', 'objective function', 'Cross Entropy');

option.batch_size = 100;
option.iteration = 1;


ratioTraining = 0.95; 

[M,N] = size(train_x);
xTraining = zeros(floor(ratioTraining*M),784);
yTraining = zeros(floor(ratioTraining*M),10);
p = randperm(M);
for i=1:floor(ratioTraining*M)
    xTraining(i,:)  = train_x(p(i),:);
    yTraining(i,:) = train_y(p(i),:);
end

xValidation = zeros(M-floor(ratioTraining*M),784);
zeros(M-floor(ratioTraining*M),10);
yValidation = [];
for i=floor(ratioTraining*M)+1:M
    xValidation(i-floor(ratioTraining*M),:)  = train_x(p(i),:);
    yValidation(i-floor(ratioTraining*M),:) = train_y(p(i),:);
end


iteration = 0;
maxAccuracy = 0;
totalAccuracy = [];
maxIteration = 100;
while(iteration<maxIteration)
    iteration = iteration +1; 
    nn = nn_train(nn,option,xTraining,yTraining);
    totalCost(iteration) = sum(nn.cost)/length(nn.cost);
   % plot(totalCost);
    [wrongs,accuracy] = nn_test(nn,xValidation,yValidation);
    totalAccuracy = [totalAccuracy,accuracy];
    if accuracy>maxAccuracy
        maxAccuracy = accuracy;
        storedNN = nn;
    end;
    cost = totalCost(iteration);
    accuracy
    cost
end;
[wrongs,accuracy] = nn_test(storedNN,test_x,test_y);

