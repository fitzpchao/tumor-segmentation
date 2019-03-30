import torch
import torch.utils.data as data
import pandas as pd
from torch.autograd import Variable as V
import cv2
import numpy as np
import random
from scipy.io import loadmat
from osgeo import gdalnumeric
import os
root_img='segimg/'
root_label='seglabel/'
def load_img(path):
    m = loadmat(path)
    img = np.array(m, dtype="float")
    img_new = img / 255.0
    return img_new

def default_loader(id):
    path_img = root_img + str(id) +".mat"
    m = loadmat(path_img)
    img = np.array(m['segimg'], dtype="float")
    img = img / 255.0
    path_label = root_label + str(id) + ".mat"
    m = loadmat(path_label)
    mask = np.array(m['seglabel'], dtype="float")
    img=np.transpose(img,[2,0,1])
    '''rot_p = random.random()
    flip_p = random.random()
    if (rot_p < 0.5):
        pass
    elif (rot_p >= 0.5):
        for k in range(3):
            img1[k, :, :] = np.rot90(img1[k, :, :])
            img2[k, :, :] = np.rot90(img2[k, :, :])
        mask = np.rot90(mask)
    if (flip_p < 0.25):
        pass
    elif (flip_p < 0.5):
        for k in range(3):
            img1[k, :, :] = np.fliplr(img1[k, :, :])
            img2[k, :, :] = np.fliplr(img2[k, :, :])
        mask = np.fliplr(mask)
    elif (flip_p < 0.75):
        for k in range(3):
            img1[k, :, :] = np.flipud(img1[k, :, :])
            img2[k, :, :] = np.flipud(img2[k, :, :])
        mask = np.flipud(mask)

    elif (flip_p < 1.0):
        for k in range(3):
            img1[k, :, :] = np.fliplr(np.flipud(img1[k, :, :]))
            img2[k, :, :] = np.fliplr(np.flipud(img2[k, :, :]))
        mask = np.fliplr(np.flipud(mask))'''
    mask=np.expand_dims(mask,axis=0)
    return  img,mask

def default_loader_val(filename,root1,root2,root3):
    path_img = root_img + str(id) +".mat"
    m = loadmat(path_img)
    img = np.array(m, dtype="float")
    img = img / 255.0
    path_label = root_label + str(id) + ".mat"
    m = loadmat(path_label)
    mask = np.array(m, dtype="float")
    img=np.transpose(img,[2,0,1])
    mask=np.expand_dims(mask,axis=0)
    return  img,mask






class ImageFolder(data.Dataset):

    def __init__(self,path):
        self.ids =  np.load(path)
        self.loader = default_loader

    def __getitem__(self, index):
        id = self.ids[index]
        img,mask = self.loader(id +1)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img,mask

    def __len__(self):
        return int(len(self.ids)/1)*1

class ImageFolder_val(data.Dataset):

    def __init__(self,path):
        self.ids =  np.load(path)
        self.loader = default_loader

    def __getitem__(self, index):
        id = self.ids[index]
        img,mask = self.loader(id +1)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img,mask,id + 1

    def __len__(self):
        return int(len(self.ids)/1)*1

