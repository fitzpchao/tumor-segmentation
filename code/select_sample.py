from scipy.io import loadmat
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

LEN=780
TEST_SIZE=0.2
LEN_TRAIN =int( LEN * (1- TEST_SIZE))
LEN_TEST = int(LEN * TEST_SIZE)
inds = np.arange(LEN)
inds_train, inds_test = train_test_split(inds, test_size=TEST_SIZE)
np.save('inds_train.npy',inds_train)
np.save('inds_test.npy',inds_test)

