import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import numpy as np
from model import FaceMobileNet, ResIRSE
from model.metric import ArcFace, CosFace
from model.loss import FocalLoss
from dataset import load_data
from config import config as conf
from sklearn.externals import joblib

def _preprocess(images: list, transform) -> torch.Tensor:
    res = []
    for img in images:
        im = Image.open(img)
        im = transform(im)
        res.append(im)
    data = torch.cat(res, dim=0)  # shape: (batch, 128, 128)
    data = data[:, None, :, :]    # shape: (batch, 1, 128, 128)
    return data


def featurize(images: list, transform, net, device) -> dict:
    """featurize each image and save into a dictionary
    Args:
        images: image paths
        transform: test transform
        net: pretrained model
        device: cpu or cuda
    Returns:
        Dict (key: imagePath, value: feature)
    """
    data = _preprocess(images, transform)

    data = data.to(device)
    net = net.to(device)
    with torch.no_grad():
        features = net(data)
    res = {img: feature for (img, feature) in zip(images, features)}
    return res

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def threshold_search(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_acc, best_th

def compute_accuracy(feature_dict, pair_list, test_root):
    with open(pair_list, 'r') as f:
        pairs = f.readlines()

    similarities = []
    labels = []
    for pair in pairs:
        img1, img2, label = pair.split()
        img1 = osp.join(test_root, img1)
        img2 = osp.join(test_root, img2)
        feature1 = feature_dict[img1].cpu().numpy()
        feature2 = feature_dict[img2].cpu().numpy()
        label = int(label)

        similarity = cosin_metric(feature1, feature2)
        similarities.append(similarity)
        labels.append(label)

    accuracy, threshold = threshold_search(similarities, labels)
    return accuracy, threshold

# Data Setup
def test():
    embedding_size = conf.embedding_size
    device = conf.device

    # Network Setup
    if conf.backbone == 'resnet':
        net = ResIRSE(embedding_size, conf.drop_ratio).to(device)
    else:
        net = FaceMobileNet(embedding_size).to(device)

    net = nn.DataParallel(net)

    # Checkpoints Setup
    checkpoints = conf.checkpoints
    weights_path = osp.join(checkpoints, conf.test_model)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.eval()
    test_root = 'D:/database/scikit_learn_data/lfw_home/lfw_funeled_cropped'


    pair_list = 'D:/database/scikit_learn_data/lfw_home/pairs.txt'
    batch_size = conf.test_batch_size
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    images = []
    labels = []
    count = 0
    pairList = lists = [[] for i in range(len(pairs))]
    for pair in pairs:
        words = pair.split()
        if len(words)==3:
            image1 = test_root+'/'+words[0]+'/'+words[0]+'_'+ str(words[1]).zfill(4)+'.jpg'
            image2 = test_root+'/'+words[0]+'/'+words[0]+'_'+ str(words[2]).zfill(4)+'.jpg'
            images.append(image1)
            images.append(image2)
            labels.append(1)
            pairList[count] = [image1,image2,1]
            count+=1
        elif len(words)==4:
            image1 = test_root+'/'+words[0]+'/'+words[0]+'_'+ str(words[1]).zfill(4)+'.jpg'
            image2 = test_root+'/'+words[2]+'/'+words[2]+'_'+ str(words[3]).zfill(4)+'.jpg'
            images.append(image1)
            images.append(image2)
            pairList[count] = [image1,image2,0]
            count += 1
    del(pairList[-1])
    size = len(images)
    groups = []
    for i in range(0, size, batch_size):
        end = min(batch_size + i, size)
        groups.append(images[i: end])

    feature_dict = dict()
    for group in groups:
        print(group)
        d = featurize(group, conf.test_transform, net, device)
        feature_dict.update(d)

    joblib.dump(feature_dict, 'feature_dictTest.pkl')

    feature_dict = joblib.load('feature_dictTest.pkl')

    similarities = []
    labels = []
    for s in pairList:
        img1 = s[0]
        img2 = s[1]
        label = s[2]
        print(img1,img2,label)
        feature1 = feature_dict[img1].cpu().numpy()
        feature2 = feature_dict[img2].cpu().numpy()

        similarity = cosin_metric(feature1, feature2)
        similarities.append(similarity)
        labels.append(label)

    accuracy, threshold = threshold_search(similarities, labels)
    print('The accuracy on the tesset is: '+str(accuracy))

    '''#Test accuracy
    pair_list = 'D:/database/scikit_learn_data/lfw_home/pairsDevTest.txt'

    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    images = []
    labels = []
    count = 0
    pairList = lists = [[] for i in range(len(pairs))]
    for pair in pairs:
        words = pair.split()
        if len(words)==3:
            image1 = test_root+'/'+words[0]+'/'+words[0]+'_'+ str(words[1]).zfill(4)+'.jpg'
            image2 = test_root+'/'+words[0]+'/'+words[0]+'_'+ str(words[2]).zfill(4)+'.jpg'
            images.append(image1)
            images.append(image2)
            labels.append(1)
            pairList[count] = [image1,image2,1]
            count+=1
        elif len(words)==4:
            image1 = test_root+'/'+words[0]+'/'+words[0]+'_'+ str(words[1]).zfill(4)+'.jpg'
            image2 = test_root+'/'+words[2]+'/'+words[2]+'_'+ str(words[3]).zfill(4)+'.jpg'
            images.append(image1)
            images.append(image2)
            pairList[count] = [image1,image2,0]
            count += 1
    del(pairList[-1])
    size = len(images)
    groups = []

    for i in range(0, size, batch_size):
        end = min(batch_size + i, size)
        groups.append(images[i: end])

    feature_dict = dict()
    for group in groups:
        print(group)
        d = featurize(group, conf.test_transform, net, device)
        feature_dict.update(d)

    joblib.dump(feature_dict, 'feature_dictTest.pkl')

    feature_dict = joblib.load('feature_dictTest.pkl')

    similarities = []
    labels = []
    for s in pairList:
        img1 = s[0]
        img2 = s[1]
        label = s[2]
        print(img1,img2,label)
        feature1 = feature_dict[img1].cpu().numpy()
        feature2 = feature_dict[img2].cpu().numpy()

        similarity = cosin_metric(feature1, feature2)
        similarities.append(similarity)
        labels.append(label)

    similarity = np.asarray(similarity)
    label = np.asarray(label)
    predictedLabel = (similarity >= threshold)
    accuracy = np.mean((predictedLabel == label).astype(int))
    print('The accuracy on the test set is: '+str(accuracy))'''


if __name__ == '__main__':
    test()

