import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision


class MLP(nn.Module):
    def __init__(self,num_hidd,nr_classes,input_dim):
        super(MLP, self).__init__()
        self.num_hidd = num_hidd
        act = nn.Sigmoid
        self.nr_classes=nr_classes
        self.input_dim = input_dim
  
        self.body = nn.Sequential(
        nn.Linear(self.input_dim,self.num_hidd),
        act())
        self.fc = nn.Sequential(nn.Linear(self.num_hidd, self.nr_classes))
    def forward(self, x):
        x = x.view(-1,self.input_dim)
        out = self.body(x)
        out = self.fc(out)
        return F.softmax(out, dim=1)

class CNN(nn.Module):
    def __init__(self,in_ch,nr_filters,k,p_size,s_size,width,C,nr_units):
        super(CNN, self).__init__()
        act = nn.Sigmoid
        self.in_ch = in_ch
        self.nr_filters = nr_filters
        self.k = k  
        self.p = p_size
        self.s = s_size
        self.d = width
        self.nr_classes = C
        self.nr_units = nr_units
        self.body = nn.Sequential(
            nn.Conv2d(self.in_ch,self.nr_filters, kernel_size=self.k, padding=self.p, stride=self.s),#k=5,p=2,s=2
            act()
        )
        self.input_dim = (self.d-self.k+2*self.p)//self.s+1
        self.fc = nn.Sequential(
            nn.Linear((self.input_dim)**2*self.nr_filters, self.nr_units),
            act(),
            nn.Linear(self.nr_units,self.nr_classes)
        )
        
    def forward(self, x):
        out1 = self.body(x)
        out2 = out1.view(out1.size(0), -1)
        out3 = self.fc(out2)
        return out3


class Simple_CNN(nn.Module):
    def __init__(self,in_ch,nr_filters,k,p_size,s_size,width,C):
        super(Simple_CNN, self).__init__()
        act = nn.Sigmoid
        self.in_ch = in_ch
        self.nr_filters = nr_filters
        self.k = k  
        self.p = p_size
        self.s = s_size
        self.d = width
        self.nr_classes = C
        self.body = nn.Sequential(
            nn.Conv2d(self.in_ch,self.nr_filters, kernel_size=self.k, padding=self.p, stride=self.s),#k=5,p=2,s=2
            act()
        )
        self.input_dim = (self.d-self.k+2*self.p)//self.s+1
        self.fc = nn.Sequential(
            nn.Linear((self.input_dim)**2*self.nr_filters,self.nr_classes))
        
    def forward(self, x):
        out1 = self.body(x)
        out2 = out1.view(out1.size(0), -1)
        out3 = self.fc(out2)
        return out3

class Original_Net(nn.Module):
    def __init__(self,in_ch,dim,C):
        super(Original_Net, self).__init__()
        act = nn.Sigmoid
        self.in_ch = in_ch
        self.dim = dim
        self.nr_classes = C
        self.body = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=1, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=1, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=1, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1, stride=1),
            act()
        )
        self.fc = nn.Sequential(
            nn.Linear(self.dim, self.nr_classes)
        )
        
    def forward(self, x):
        out1 = self.body(x)
        out2 = out.view(out1.size(0), -1)
        out3 = self.fc(out2)
        return out3

class CNN_1stNet(nn.Module):
    def __init__(self,in_ch,nr_filters,k,p_size,s_size,C):    
        super(CNN_1stNet, self).__init__()
        act = nn.Sigmoid
        self.in_ch = in_ch
        self.k = k
        self.nr_filters = nr_filters
        self.p = p_size
        self.s = s_size
#         self.d = width
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_ch,self.nr_filters, kernel_size=self.k, padding=self.p, stride=self.s),#k=5,p=2,s=2
            act()
        )        
    def forward(self, x):
        out1 = self.cnn(x)
        return out1

class CNN_2ndNet(nn.Module):
    def __init__(self,input_dim,nr_units,C):
        super(CNN_2ndNet,self).__init__()
        act = nn.Sigmoid
        self.input_dim = input_dim
        self.nr_units = nr_units
        self.C = C
        self.body = nn.Sequential(
        	nn.Linear(self.input_dim, self.nr_units),
        	act(),
        	nn.Linear(self.nr_units,self.C))

    def forward(self,x):
        x = x.view(-1,self.input_dim)
        out = self.body(x)
        return F.softmax(out, dim=1)
   

