
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms

tt = transforms.ToPILImage()
def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))




def recon_first_net(out,first_net,recon_data,epochs,lamda,m=100,opt=None):
    losses = []
    history = {}
    for l in range(len(recon_data)):
        history['image'+str(l)]=[]
    optimizer=torch.optim.Adam([recon_data])#LBFGS
    for itrs in range(epochs):
        def closure():
            optimizer.zero_grad()
            dummy_out = first_net(recon_data)
            grad_diff = ((out.view(-1,1)-dummy_out.view(-1,1))**2).sum()
            l = 0
            if opt == 'orth':
                for i in range(1,len(recon_data)):
                    l+=torch.matmul(recon_data[0].reshape(1,-1),recon_data[i].reshape(-1,1))**2*(lamda*(0.9**((itrs+1)//m)))
            if opt == 'l1':
                l = torch.sum(torch.abs(recon_data))*0.0001#0.001
            if opt == 'l2':
                l = torch.norm(recon_data)*lamda*(0.9**(itrs//500))
            ll = grad_diff+l
            ll.backward(retain_graph=True)
            return ll
        optimizer.step(closure)
        if itrs%100==0:
            dist = 0
            current_loss=closure()
            losses.append(current_loss)
            print('itr:{},loss:{:.7f}'.format(itrs,current_loss.item()))
        for l in range(len(recon_data)):
            history['image'+str(l)].append(tt(recon_data[l].cpu()))    
    return history,recon_data,losses



def recon_l2(net,criterion,recon_data,recon_label,original_dy_dx,epochs,lamda,m,opt=None):
    losses = []
    history = {}
    for l in range(len(recon_data)):
        history['image'+str(l)]=[]
    optimizer=torch.optim.Adam([recon_data,recon_label])#LBFGS
    for itrs in range(epochs):
        def closure():
            optimizer.zero_grad()
            pred = net(recon_data)
            dummy_onehot_label = F.softmax(recon_label)
            dummy_loss = criterion(pred,dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss,net.parameters(),create_graph=True)
            grad_diff = 0
            grad_count = 0
            for gx,gy in zip(dummy_dy_dx,original_dy_dx):
                grad_diff+=((gx-gy)**2).sum()
                grad_count+=gx.nelement()
            l = 0
            if opt == 'orth':
                # print('=> orthogonality regularizer')
                for j in range(len(recon_data)):
                    for i in range(j+1,len(recon_data)):
                        l+=torch.matmul(recon_data[j].reshape(1,-1),recon_data[i].reshape(-1,1))**2
                l = l*(lamda*(0.9**((itrs+1)//m)))
            if opt == 'l1':
                l = torch.sum(torch.abs(recon_data))*lamda*(0.9**(itrs//m))#0.001
            if opt == 'l2':
                l = torch.norm(recon_data)*lamda*(0.9**(itrs//m))
            ll = grad_diff+l
            ll.backward()
            return ll
        optimizer.step(closure)
        if itrs%50==0:
            current_loss=closure()
            losses.append(current_loss)
            print('itr:{},loss:{:.7f}'.format(itrs,current_loss.item()))
        for l in range(len(recon_data)):
            history['image'+str(l)].append(tt(recon_data[l].cpu()))
    return history,recon_data,losses
