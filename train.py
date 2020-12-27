# -*- coding: utf-8 -*-
from utils.agent import *
from utils.dataset import *



from IPython.display import clear_output

import sys
import traceback
import sys
import os


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

# Done = ['cat', 'bird', 'motorbike', 'diningtable', 'train' ]
classes = [ 'tvmonitor', 'bus', 'horse', 'car', 'pottedplant', 'person', 'chair', 'boat', 'bottle', 'bicycle', 'dog', 'aeroplane', 'cow', 'sheep', 'sofa']

agents_per_class = {}
datasets_per_class = sort_class_extract([train_loader2007, train_loader2012])
for key in classes:
    agents_per_class[key] = Agent(key, alpha=0.2, num_episodes=15, load=False)
    agents_per_class[key].train(datasets_per_class[key])
