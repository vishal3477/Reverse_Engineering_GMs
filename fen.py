from torchvision import datasets, models, transforms
import os
import torch
from torch.autograd import Variable
from skimage import io
from scipy import fftpack
import numpy as np
from torch import nn
import datetime
import encoder_deepfake
import fen
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn import metrics
import argparse

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0,n,None) 
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n,None,None)
                  for i in range(X.dim()))
    #print(axis,n,f_idx,b_idx)
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front],axis)

def fftshift(real, imag):
    for dim in range(1, len(real.size())):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return real, imag



class DnCNN(nn.Module):
    def __init__(self, num_layers=31, num_features=64):
        super(DnCNN, self).__init__()
        layers = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(num_features, 3, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        y = inputs
        residual = self.layers(y)
        residual_1 = residual.clone()
        
        residual_gray=0.299*residual_1[:,0,:,:].clone()+0.587*residual_1[:,1,:,:].clone()+0.114*residual_1[:,2,:,:].clone()
        
        thirdPart_fft_1=torch.rfft(residual_gray, signal_ndim=2, onesided=False)
        
        thirdPart_fft_1_orig=thirdPart_fft_1.clone()
        
        thirdPart_fft_1[:,:,:,0],thirdPart_fft_1[:,:,:,1]=fftshift(thirdPart_fft_1[:,:,:,0],thirdPart_fft_1[:,:,:,1])
        thirdPart_fft_1=torch.sqrt(thirdPart_fft_1[:,:,:,0]**2+thirdPart_fft_1[:,:,:,1]**2)
        n=25
        (_,w,h)=thirdPart_fft_1.shape
        half_w, half_h = int(w/2), int(h/2)
        thirdPart_fft_2=thirdPart_fft_1[:,half_w-n:half_w+n+1,half_h-n:half_h+n+1].clone()
        thirdPart_fft_3=thirdPart_fft_1.clone()
        thirdPart_fft_3[:,half_w-n:half_w+n+1,half_h-n:half_h+n+1]=0
        max_value=torch.max(thirdPart_fft_3)
        thirdPart_fft_4=thirdPart_fft_1.clone()
        thirdPart_fft_4=torch.transpose(thirdPart_fft_4,1,2)
        return thirdPart_fft_1,thirdPart_fft_2, max_value, thirdPart_fft_1_orig,residual, thirdPart_fft_4, residual_gray