import cv2
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from read_data_single import ImageFolder_val
from unet_attention import UNet
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

ROOT_PRED='results_train'
VAL_LIST="inds_train.npy"
VAL_LIST="inds_test.npy"
ROOT_PRED='results_test'



def get_dataloader(batch_size=1):
    '''mytransform = transforms.Compose([
        transforms.ToTensor()])'''

    # torch.utils.data.DataLoader
    validation_loader = torch.utils.data.DataLoader(
        ImageFolder_val(VAL_LIST
                    ),
        batch_size=batch_size, shuffle=False)
    return validation_loader

def main():
    #torch.backends.cudnn.benchmark = True
    #model=DinkNet34(num_classes=1)
    model = UNet(n_channels=1)
    model=torch.nn.DataParallel(model)
    model=model.cuda()
    model.load_state_dict(torch.load("checkpoints/exp4/ model_65.pkl")['weight'])
    model.eval()
    matrixs=np.zeros([2,2],np.float32)
    data_loader = get_dataloader(1)


    for data,target,id in tqdm(data_loader):
        data = data.cuda()
        output = model(data)
        output = F.sigmoid(output).cpu()
        output = output.data.numpy()
        output[output>=0.5] = 1
        output[output<0.5] = 0
        target=target.data.numpy()
        pred=np.squeeze(output).copy()
        pred[pred==1]=255
        target=np.squeeze(target)
        target[target>0]=255
        if not os.path.exists(ROOT_PRED):
            os.makedirs(ROOT_PRED)
        cv2.imwrite(ROOT_PRED +'/'+ str(id.data.numpy()[0]) + 'pred_.png',pred)
        cv2.imwrite(ROOT_PRED + '/' + str(id.data.numpy()[0]) + 'gt_.png', target)
        target[target>0]=1
        target=np.reshape(target,[-1])
        output=np.reshape(output,[-1])
        target=target.astype(np.int8)
        output=output.astype(np.int8)
        labels=list(set(np.concatenate((target,output),axis=0)))
        matrixs_temp = np.zeros([2, 2], np.float32)
        print(labels)
        if(labels == [0]):
            matrixs[0, 0] +=confusion_matrix(target,output)[0,0]
            matrixs_temp[0,0] =confusion_matrix(target,output)[0,0]
        elif (labels == [1]):
            matrixs[1, 1] += confusion_matrix(target, output)[0, 0]
            matrixs_temp[1,1] =confusion_matrix(target,output)[0,0]
        else:
            matrixs += confusion_matrix(target,output)
            matrixs_temp=confusion_matrix(target,output)
        print(matrixs_temp)


    confusion_matrixs=pd.DataFrame(matrixs)
    confusion_matrixs.to_csv('confusion_matrix_attention_train.csv',header=None,index=None)





if __name__ == '__main__':
    main()
