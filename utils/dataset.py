import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

batch_size = 32
PATH="./datasets/"


class CustomRotation(object):
    """
        Fournit une classe trnaform qui remet les images dans la bonne orientation ( car mal orientées orignellement dans le jeu de données )
    """
    def __call__(self, image):
        return image.transpose(0, 2).transpose(0, 1)



def get_transform(train):
    """
        Permettant la préparation d'une fonction normalisation + le redimensionnement des images du jeu de données.
    """
    base_size = 520
    crop_size = 480

    min_size = int((0.5 if train else 1.0) * base_size)
    max_size = int((2.0 if train else 1.0) * base_size)
    transf = []
    transf.append( transforms.ToTensor())
    transf.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transf)
    
def make_image_transform(image_transform_params: dict,
                         transform: object):
    resize_image = image_transform_params['image_mode']
    if resize_image == 'none':
        preprocess_image = None
    elif resize_image == 'shrink':
        preprocess_image = transforms.Resize((image_transform_params['output_image_size']['width'],
                                              image_transform_params['output_image_size']['height']))
    elif resize_image == 'crop':
        preprocess_image = transforms.CenterCrop((image_transform_params['output_image_size']['width'],
                                                  image_transform_params['output_image_size']['height']))

    if preprocess_image is not None:
        if transform is not None:
            image_transform = transforms.Compose([preprocess_image, transform])
        else:
            image_transform = preprocess_image
    else:
        image_transform = transform
    return image_transform


def read_voc_dataset(download=True, year='2007'):
    """
        Fonction qui récupére les dataloaders de validation et de train du jeu de données PASCAL VOC selon l'année d'entrée

    """
    T = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                             #CustomRotation()
                            ])
    voc_data =  torchvision.datasets.VOCDetection(PATH, year=year, image_set='train', 
                        download=download, transform=T)
    train_loader = DataLoader(voc_data,shuffle=True)

    voc_val =  torchvision.datasets.VOCDetection(PATH, year=year, image_set='val', 
                        download=download, transform=T)
    val_loader = DataLoader(voc_val,shuffle=False)

    return voc_data, voc_val

def get_images_labels(dataloader):
    """
        Récupére séparemment les images et labels du dataloader
    """
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    return images, labels



"""
Fonctions permettant la récupération et lecture du jeu de données SB de Pytorch
"""
class NoisySBDataset():
    def __init__(self, path, image_set="train", transforms = None, download=True):
        super().__init__()
        self.transforms = transforms
        self.dataset = torchvision.datasets.SBDataset(root=path,
                                                      image_set=image_set,
                                                      download=download)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):  # a[x] for calling a.__getitem__(x)
        img, truth = self.dataset[idx]
        if self.transforms:
            img = self.transforms(img)
        return (img, truth)


def read_sbd_dataset(batch_size, download=True):
    """
        Lecture et normalisation du jeu de données SB.
    """
    T = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor()
                            ])
    voc_data =  NoisySBDataset(PATH, image_set='train', 
                        download=download, transforms=T)
    train_loader = DataLoader(voc_data, batch_size=32,shuffle=False,  collate_fn=lambda x: x)

    voc_val =  NoisySBDataset(PATH, image_set='val', 
                        download=download, transforms=T)
    val_loader = DataLoader(voc_val, batch_size=32,shuffle=False,  collate_fn=lambda x: x)

    return train_loader, val_loader
