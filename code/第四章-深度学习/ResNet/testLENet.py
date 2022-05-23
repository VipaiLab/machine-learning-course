import torch
from torch import nn, optim
from torch.nn import init
from torchvision import datasets, transforms
from torch.utils import data
import sys

use_gpu = torch.cuda.is_available()
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(), #也可 nn.Sigmoid()
            nn.MaxPool2d(2,2), #也可nn.AvgPool2d(2,2)
            nn.Conv2d(6, 16, 5),
            nn.ReLU(), #也可 nn.Sigmoid()
            nn.MaxPool2d(2,2) #也可nn.AvgPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(),#也可 nn.Sigmoid()
            nn.Linear(120, 84),
            nn.ReLU(),#也可 nn.Sigmoid()
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

net = LeNet()
if (use_gpu):
    net.cuda()

batch_size = 100
trainingdataset = datasets.ImageFolder(root = 'datasets/MNIST/train', transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]))
train_iter = data.DataLoader(trainingdataset, batch_size = batch_size, shuffle = True )

if (use_gpu):
    criterion = torch.nn.CrossEntropyLoss().cuda()
else:
    criterion = torch.nn.CrossEntropyLoss()

#optimizer = torch.optim.SGD(net.parameters(),lr=1e-3)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9,0.999), eps=1e-08, weight_decay=0)
num_epochs = 50

batch_size = 200
testingdataset = datasets.ImageFolder(root='datasets/MNIST/test', transform=transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]))
test_iter = data.DataLoader(testingdataset, batch_size=batch_size, shuffle=False)
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
        torch.save(net, 'model.pkl')

print('Finished Training')

net = torch.load('model.pkl')
batch_size = 200
testingdataset = datasets.ImageFolder(root='datasets/MNIST/test', transform=transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]))
test_iter = data.DataLoader(testingdataset, batch_size=batch_size, shuffle=False)


recognitionRate = 0
totalNumber = 0
for inputs, labels in test_iter:
    outputs = net(inputs)
    recognitionRate += (outputs.argmax(dim=1) == labels).float().sum().item()
    totalNumber += labels.shape[0]

recognitionRate = 1.0*recognitionRate/totalNumber
print('The recognition rate is %f' % recognitionRate)
