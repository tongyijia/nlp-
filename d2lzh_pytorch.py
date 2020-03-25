from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torchvision
from IPython import display
from torch import nn
from torch.nn import init
import time

def use_svg_display():
    #用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) #样本的读取顺序随机
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)]) #最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)
        
def linreg(X, w, b):
    return torch.mm(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2
       
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size
        
        
#将数值标签转成相应的⽂文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankleboot']

    return [text_labels[int(i)] for i in labels]


#在⼀行里画出多张图像和对应标签的函数
def show_fashion_mnist(images, labels):
    use_svg_display()
    _,figs = plt.subplots(1,len(images),figsize=(12,12))
    
    for f, img, lbl in zip(figs,images,labels):
        f.imshow(img.view((28,28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
    
def load_data_fashion_mnist(batch_size, resize=None,root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into
memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
      
    trans.append(torchvision.transforms.ToTensor())
    
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root,train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root,train=False, download=True, transform=transform)
    train_iter = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_iter, test_iter

# def evaluate_accuracy(data_iter, net):
#     acc_sum, n = 0.0, 0
#     for X,y in data_iter:
#         if isinstance(net, torch.nn.Module):
#             net.eval() #评估模式，这会关闭dropout
            
#             acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            
#             net.train()#改回训练模式
#         else:
#             if ('is_training' in net.__code__.co_varnames):
#                 acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
#             else:
#                 acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
#         n += y.shape[0]
#     return acc_sum / n

def evaluate_accuracy(data_iter, net, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() #评估模式，这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else: #自定义模型，3，13节之后不会用到，不考虑GPU
                if('is_training' in net.__code__.co_varnames): #如果有is_training这个参数
                    #将is_training设置为False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
            
    return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0., 0., 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            
            #梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()
                
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % \
              (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print('training on',device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' % \
              (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    def forward(self,x):
        return x.view(x.shape[0],-1)
    
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
        
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i+h, j:j+w] * K).sum()
    return Y
