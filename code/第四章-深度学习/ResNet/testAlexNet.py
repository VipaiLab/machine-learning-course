import time
import torch
from torch import nn, optim
from torch.nn import init
from torchvision import datasets, transforms
from torch.utils import data
import sys

use_gpu = torch.cuda.is_available()
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), #in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3,2), #kernel_size, stride
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),#也可 nn.Sigmoid()
            nn.Linear(4096, 4096),
            nn.ReLU(),#也可 nn.Sigmoid()
            nn.Linear(4096, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

net = AlexNet()
if (use_gpu):
    net.cuda()

batch_size = 100
dataset = datasets.ImageFolder(root = 'datasets/MNIST/train', transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                                                                      transforms.Resize([227, 227]),
                                                                                                      transforms.RandomRotation(5),
                                                                                                      transforms.ToTensor()]))
train_iter = data.DataLoader(dataset, batch_size = batch_size, shuffle = True )
if (use_gpu):
    criterion = torch.nn.CrossEntropyLoss().cuda()
else:
    criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9,0.999), eps=1e-08, weight_decay=0)
num_epochs = 1

batch_size = 200
dataset = datasets.ImageFolder(root='datasets/MNIST/test', transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                                                                      transforms.Resize([227, 227]),
                                                                                                      transforms.RandomRotation(5),
                                                                                                      transforms.ToTensor()]))
test_iter = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
maxRecognitionRate = 0
for epoch in range(num_epochs):
    running_loss = 0.0
    number = 0
    for inputs, labels in train_iter:
        if (use_gpu):
            inputs = inputs.cuda()
            labels = labels.cuda()
        number+=1
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if number % 10 == 9:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, number + 1, running_loss / 10))
            running_loss = 0.0

    recognitionRate = 0
    totalNumber = 0
    for inputs, labels in test_iter:
        if (use_gpu):
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = net(inputs)
        recognitionRate += (outputs.argmax(dim=1) == labels).float().sum().item()
        totalNumber += labels.shape[0]
    recognitionRate = 1.0 * recognitionRate / totalNumber
    print('The recognition rate is %f' % recognitionRate)
    if recognitionRate>maxRecognitionRate:
        torch.save(net, 'modelAlexnet.pkl')

import numpy as np
np.set_printoptions(threshold=np.inf)
file = open('parameters.txt', 'w')
for name, param in net.named_parameters():
    print(name, list(param), file=file)
file.close()
