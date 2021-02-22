# -*- coding: utf-8 -*-
from utils.agent import *
from utils.dataset import *



from IPython.display import clear_output

import sys
import traceback
import sys
import os
import tqdm.notebook as tq
import seaborn as sns

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

#  'cat', 'dog', 'bird', 'motorbike','diningtable', 'train', 'tvmonitor', 'bus', 'horse', 'car',  Deja fait

classes = [ 'pottedplant', 'person', 'chair', 'boat', 'bottle', 'bicycle', 'aeroplane', 'cow', 'sheep', 'sofa']


agents_per_class = {}
datasets_per_class = sort_class_extract([train_loader2007, train_loader2012])
datasets_eval_per_class = sort_class_extract([val_loader2007, val_loader2012])
for i in range(len(classes)):
    classe = classes[i]
    print("Classe "+str(classe)+"...")
    agents_per_class[classe] = Agent(classe, alpha=0.2, num_episodes=15, load=False)
    agents_per_class[classe].train(datasets_per_class[classe])
    del agents_per_class[classe]
    torch.cuda.empty_cache()


