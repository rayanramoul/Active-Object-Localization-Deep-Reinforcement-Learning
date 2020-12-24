# -*- coding: utf-8 -*-
import torchvision
#import torchvision.datasets.SBDataset as sbd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


from utils.agent import *
from utils.dataset import *
from utils.models import *
from utils.tools import *


import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from IPython.display import clear_output

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch.optim as optim
import cv2 as cv
import sys
from torch.autograd import Variable
import traceback
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#!pip3 install torch==1.5.1 torchvision==0.6.1 -f https://download.pytorch.org/whl/cu92/torch_stable.html
try:
    if 'google.colab' in str(get_ipython()):
        from google.colab import drive
        drive.mount('/content/gdrive')
        LOAD = True
        SAVE_MODEL_PATH = '/content/gdrive/MyDrive/models/' + 'q_network'
    else:
        LOAD = False
        SAVE_MODEL_PATH = "./models/q_network"
except NameError:
        LOAD = False
        SAVE_MODEL_PATH = "./models/q_network"
batch_size = 32
PATH="./datasets/"



train_loader2012, val_loader2012 = read_voc_dataset(download=LOAD, year='2012')
train_loader2007, val_loader2007 = read_voc_dataset(download=LOAD, year='2007')


classes = ['cat', 'bird', 'motorbike', 'diningtable', 'train', 'tvmonitor', 'bus', 'horse', 'car', 'pottedplant', 'person', 'chair', 'boat', 'bottle', 'bicycle', 'dog', 'aeroplane', 'cow', 'sheep', 'sofa']

agents_per_class = {}
datasets_per_class = sort_class_extract([train_loader2007, train_loader2012])
for key in classes:
    agents_per_class[key] = Agent(key, alpha=0.02, num_episodes=25, load=False)
    agents_per_class[key].train(datasets_per_class[key])
