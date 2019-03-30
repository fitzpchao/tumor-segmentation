import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from read_data_single import *
from unet_attention import UNet
from utils_loss import Trainer

BATCH_SIZE=8
TRAIN_LIST="inds_train.npy"
VAL_LIST="inds_test.npy"
SAVE_PATH='checkpoints/exp4'
def get_dataloader(batch_size):
    '''mytransform = transforms.Compose([
        transforms.ToTensor()])'''

    # torch.utils.data.DataLoader
    train_loader = torch.utils.data.DataLoader(
        ImageFolder(TRAIN_LIST
                      ),
        batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(
        ImageFolder(VAL_LIST
                      ),
        batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader

def main(batch_size):
    train_loader, test_loader = get_dataloader(batch_size)
    #model= DinkNet34(num_classes=1)
    model=UNet(n_channels=1)

    #optimizer = optim.SGD(params=model.parameters(), lr=1e-1, momentum=0.9,weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, 80, 0.1)
    optimizer=optim.Adam(params=model.parameters())
    trainer = Trainer(model, optimizer, nn.BCELoss ,save_freq=1,save_dir=SAVE_PATH)
    trainer.loop(400, train_loader, test_loader)


if __name__ == '__main__':
    main(BATCH_SIZE)