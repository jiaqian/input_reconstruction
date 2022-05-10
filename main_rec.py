import numpy as np
from pprint import pprint
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from sklearn.utils import shuffle
import argparse
import os
from network import *
from utils import *
from os import listdir
from os.path import isfile, join
torch.manual_seed(50)
torch.cuda.manual_seed(50)

parser = argparse.ArgumentParser(description='reconstruction')
parser.add_argument('--lamda',type=float,default=1)
parser.add_argument('--m',type=int,default=200)
parser.add_argument('--epochs',type=int,default=20000)
parser.add_argument('--kernel_size',type=int,default=5)
parser.add_argument('--padding_size',type=int,default=2)
parser.add_argument('--stride_size',type=int,default=2)
parser.add_argument('--dataset_name',type=str,default='mnist')
parser.add_argument('--add',type=int,default=0)
parser.add_argument('--bth',type=int,default=4)
parser.add_argument('--net',type=str,default='MLP')
parser.add_argument('--nr_filters',type=int,default=4)
# parser.add_argument('--method',type=str,default='iterative',required=True)
parser.add_argument('--save',type=bool,default=False)
parser.add_argument('--regularizer',type=str,default='none')
args = parser.parse_args()



tt = transforms.ToPILImage()
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

def main():
    if args.dataset_name=='mnist':
        print('*'*10+'mnist'+'*'*10)
        C = 10
        input_dim = 1024
        in_ch = 1
        width = 32
        dim = 300
        data = datasets.MNIST("data/", download=True)
        tp = transforms.Compose([
        	transforms.Resize(32),
        	transforms.CenterCrop(32),
        	transforms.ToTensor()])
    elif args.dataset_name=='cifar100':
        print('*'*10+'mnist'+'*'*10)
        C = 100
        input_dim = 1024*3
        in_ch = 3
        width = 32
        dim = 768
        data = datasets.CIFAR100("data/", download=True)
        tp = transforms.Compose([
    	    transforms.Resize(32),
    	    transforms.CenterCrop(32),
    	    transforms.ToTensor()])
    else:
        raise Exception("Please use either cifar100 or mnist")
    img_index = range(args.bth)
    x = torch.stack([tp(data[i][0]).to(device) for i in img_index])
    label = torch.stack([torch.Tensor([data[i][1]]).long().to(device) for i in img_index])
    label = label.view(args.bth, )
    # print(label)
    gt_onehot_label = label_to_onehot(label, num_classes=C)
    criterion = cross_entropy_for_onehot
    if args.net=='MLP':
        num_hidd = args.bth + args.add
        net = MLP(num_hidd,C,input_dim)    
    elif args.net=='CNN':
        num_hidd = args.bth + args.add
        net = CNN(in_ch,args.nr_filters,args.kernel_size,args.padding_size,args.stride_size,width,C,num_hidd)
    elif args.net=='Simple_CNN':
        net = Simple_CNN(in_ch,args.nr_filters,args.kernel_size,args.padding_size,args.stride_size,width,C)
    elif args.net=='Original_Net':
        net = Original_Net(in_ch,dim,C)
    net.apply(weights_init)
    pred = net(x)
    loss = criterion(pred,gt_onehot_label)
    gradients = torch.autograd.grad(loss, net.parameters())
    dy_dx = list((_.detach().clone() for _ in gradients)) 
   
    if args.bth==1:
        if args.net=='MLP':
        	recon_data = dy_dx[0]/dy_dx[1]
        	if args.save:
        	   np.save(args.dataset_name+'_oneinstance_mlp_rec.npy',recon_data.cpu().detach().numpy()) 
        elif args.net == 'CNN':
            rec_cnn_out = dy_dx[2]/dy_dx[3]
            first_net = CNN_1stNet(in_ch,args.nr_filters,args.kernel_size,args.padding_size,args.stride_size,C).to(device)
            first_net.apply(weights_init)
            out_of_conv = first_net(x)
            recon_data = torch.rand(x.size()).to(device).requires_grad_(True)
            history,recon_data,losses = recon_first_net(out_of_conv,first_net,recon_data,args.epochs,args.lamda,m=100,opt=args.regularizer)
            if args.save:
                np.save(args.dataset_name+'_epochs_'+str(args.epochs//1000)+'k_cnn_twosteps_rec.npy',recon_data.cpu().detach().numpy())
                np.savetxt(args.dataset_name+'_epochs_'+str(args.epochs//1000)+'k_cnn_twosteps_grad_loss.txt',losses)
    else:
        recon_data = torch.rand(x.size()).to(device).requires_grad_(True)
        recon_label = torch.rand(gt_onehot_label.size()).to(device).requires_grad_(True)
        history,recon_data,losses = recon_l2(net,criterion,recon_data,recon_label,dy_dx,args.epochs,args.lamda,args.m,opt=args.regularizer)
        if args.save:
            np.save(args.dataset_name+'_epochs_'+str(args.epochs//1000)+'k_'+args.net+'_batch_rec.npy',recon_data.cpu().detach().numpy())
            np.savetxt(args.dataset_name+'_epochs_'+str(args.epochs//1000)+'k_'+args.net+'_batch_grad_loss.txt',losses)




if __name__ == '__main__':
    main()