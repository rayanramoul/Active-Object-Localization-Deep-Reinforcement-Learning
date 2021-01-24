use_cuda = False
import torch
import torch.nn as nn
from collections import namedtuple
import torchvision.transforms as transforms

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


use_cuda = True
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
if use_cuda:
    criterion = nn.MSELoss().cuda()   
else:
    criterion = nn.MSELoss()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
           # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))  #  numbers here need to be adjusted in future
])