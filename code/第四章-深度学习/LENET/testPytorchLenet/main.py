import torch
from torch.utils.data import DataLoader
import scipy.misc
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import optim

'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding = 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def train():
    learning_rate = 1e-3
    batch_size = 100
    epoches = 50
    lenet = LeNet()
    trans_img = transforms.Compose([transforms.ToTensor()])
    trainset = MNIST('./data', train=True, transform=trans_img, download=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
    criterian = nn.CrossEntropyLoss(reduction='sum')  #loss
    optimizer = optim.SGD(lenet.parameters(), lr=learning_rate)  #optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #lenet.to("cpu")

    for i in range(epoches):
        running_loss = 0.
        running_acc = 0.
        for (img, label) in trainloader:
            optimizer.zero_grad()  # 求梯度之前对梯度清零以防梯度累加
            output=lenet(img)
            loss=criterian(output,label)
            loss.backward()    #loss反传存到相应的变量结构当中
            optimizer.step()   #使用计算好的梯度对参数进行更新
            running_loss+=loss.item()
            #print(output)
            _,predict=torch.max(output,1)
            correct_num=(predict==label).sum()
            running_acc+=correct_num.item()

        running_loss/=len(trainset)
        running_acc/=len(trainset)
        print("[%d/%d] Loss: %.5f, Acc: %.2f" % (i + 1, epoches, running_loss,100 * running_acc))

    return lenet


def test(lenet):
    batch_size = 100
    trans_img = transforms.Compose([transforms.ToTensor()])
    testset = MNIST('./data', train=False, transform=trans_img, download=True)
    testloader = DataLoader(testset, batch_size, shuffle=False, num_workers=10)
    running_acc = 0.
    for (img, label) in testloader:
        output = lenet(img)
        _, predict = torch.max(output, 1)
        correct_num = (predict == label).sum()
        running_acc += correct_num.item()
    running_acc /= len(testset)
    return running_acc


if __name__ == '__main__':
    lenet = train()
    torch.save(lenet, 'lenet.pkl') # save model

    lenet = torch.load('lenet.pkl') # load model
    test_acc = test(lenet)
    print("Test Accuracy:Loss: %.2f" % test_acc)




