import torch  # Import main library
from torch.utils.data import DataLoader  # Main class for threaded data loading
import torch.nn.functional as func
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix

train_data = np.load('../base_dades_xarxes_neurals/train.npy')
val_data = np.load('../base_dades_xarxes_neurals/val.npy')
