trainFolder = 'MNIST DATASET\train';
DIR = dir(trainFolder);
DIR(1:2) = [];

numberOfFiles = zeros(1,10);
for i = 1:length(DIR)
    index = str2num(DIR(i).name);
    subDIR = dir([trainFolder,'\',DIR(i).name]);
    subDIR(1:2) = [];
    numberOfFiles(index+1) = length(subDIR);
end

train_x = zeros(sum(numberOfFiles), 784);
train_y = zeros(sum(numberOfFiles), 10);

numFiles = 0;
vector = zeros(784,1);
for i = 1:length(DIR)
    label = zeros(10,1);
    index = str2num(DIR(i).name);
    label(index+1) = 1;
    
    subDIR = dir([trainFolder,'\',DIR(i).name]);
    subDIR(1:2) = [];
    for j = 1:length(subDIR)
        image = imread([trainFolder,'\',DIR(i).name,'\',subDIR(j).name]);
        for u = 1:28
            for v = 1:28
                vector(28*(u-1)+v) = image(u,v);
            end
        end
        numFiles = numFiles+1;
        train_x(numFiles,:) = vector;
        train_y(numFiles,:) = label;
    end
end

testFolder = 'MNIST DATASET\test';
DIR = dir(testFolder);
DIR(1:2) = [];

numberOfFiles = zeros(1,10);
for i = 1:length(DIR)
    index = str2num(DIR(i).name);
    subDIR = dir([testFolder,'\',DIR(i).name]);
    subDIR(1:2) = [];
    numberOfFiles(index+1) = length(subDIR);
end

test_x = zeros(sum(numberOfFiles), 784);
test_y = zeros(sum(numberOfFiles), 10);

numFiles = 0;
vector = zeros(784,1);
for i = 1:length(DIR)
    label = zeros(10,1);
    index = str2num(DIR(i).name);
    label(index+1) = 1;
    
    subDIR = dir([testFolder,'\',DIR(i).name]);
    subDIR(1:2) = [];
    for j = 1:length(subDIR)
        image = imread([testFolder,'\',DIR(i).name,'\',subDIR(j).name]);
        for u = 1:28
            for v = 1:28
                vector(28*(u-1)+v) = image(u,v);
            end
        end
        numFiles = numFiles+1;
        test_x(numFiles,:) = vector;
        test_y(numFiles,:) = label;
    end
end

save mnist_uint8 train_x train_y test_x test_y
    