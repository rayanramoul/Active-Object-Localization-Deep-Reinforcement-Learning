
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from config import *
import random

classes = ['cat', 'bird', 'motorbike', 'diningtable', 'train', 'tvmonitor', 'bus', 'horse', 'car', 'pottedplant', 'person', 'chair', 'boat', 'bottle', 'bicycle', 'dog', 'aeroplane', 'cow', 'sheep', 'sofa']

def sort_class_extract(datasets):    
    """
        Permet le tri par classes d'un jeu de données, parcours l'ensemble des objets d'une image et si il trouve un objet d'une classe
        il l'ajoute au jeu de données de celle-ci. 
        Entrée :
            - Jeu de données. 
        Sortie :
            - Dictionnaire de jeu de données. ( clés : Classes, valeurs : Toutes les données de cette classe )
    """
    datasets_per_class = {}
    for j in classes:
        datasets_per_class[j] = {}

    for dataset in datasets:
        for i in dataset:
            img, target = i
            classe = target['annotation']['object'][0]["name"]
            filename = target['annotation']['filename']

            org = {}
            for j in classes:
                org[j] = []
                org[j].append(img)
            for i in range(len(target['annotation']['object'])):
                classe = target['annotation']['object'][i]["name"]
                org[classe].append(  [   target['annotation']['object'][i]["bndbox"], target['annotation']['size']   ]  )
            
            for j in classes:
                if len( org[j] ) > 1:
                    try:
                        datasets_per_class[j][filename].append(org[j])
                    except KeyError:
                        datasets_per_class[j][filename] = []
                        datasets_per_class[j][filename].append(org[j])       
    return datasets_per_class


def show_new_bdbox(image, labels, color='r', count=0):
    """
        Fonction pour la visualisation des boites englobantes directement sur l'image.
    """
    xmin, xmax, ymin, ymax = labels[0],labels[1],labels[2],labels[3]
    fig,ax = plt.subplots(1)
    ax.imshow(image.transpose(0, 2).transpose(0, 1))

    width = xmax-xmin
    height = ymax-ymin
    rect = patches.Rectangle((xmin,ymin),width,height,linewidth=3,edgecolor=color,facecolor='none')
    ax.add_patch(rect)
    ax.set_title("Iteration "+str(count))
    plt.savefig(str(count)+'.png', dpi=100)


def extract(index, loader):
    """
        A partir du dataloader extrait ( et sépare ) les images et les boites englobantes vérité terrain
        et adaptent les coordonnées par rapport aux nouvelles tailles d'images.
    """
    extracted = loader[index]
    ground_truth_boxes =[]
    for ex in extracted:
        img = ex[0]
        bndbox = ex[1][0]
        size = ex[1][1]
        xmin = ( float(bndbox['xmin']) /  float(size['width']) ) * 224
        xmax = ( float(bndbox['xmax']) /  float(size['width']) ) * 224

        ymin = ( float(bndbox['ymin']) /  float(size['height']) ) * 224
        ymax = ( float(bndbox['ymax']) /  float(size['height']) ) * 224

        ground_truth_boxes.append([xmin, xmax, ymin, ymax])
    return img, ground_truth_boxes




def voc_ap(rec, prec, voc2007=False):
    """
        Calcul de l'AP et du Recall. Si voc2007 est vraie on utilise alors la mesure préconisé par le papier de PASCAL VOC 2007 ( méthode des 11 points )
    """
    if voc2007:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def prec_rec_compute(bounding_boxes, gt_boxes, ovthresh):
    """
        Calcul de précision et recall grâce à l'Intersection/Union et selon le threshold entre les vérités terrains et les prédictions.
    """
    nd = len(bounding_boxes)
    npos = nd
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    d = 0

    for index in range(len(bounding_boxes)):
        box1 = bounding_boxes[index]
        box2 = gt_boxes[index][0]
        x11, x21, y11, y21 = box1[0], box1[1], box1[2], box1[3]
        x12, x22, y12, y22 = box2[0], box2[1], box2[2], box2[3]
        
        yi1 = max(y11, y12)
        xi1 = max(x11, x12)
        yi2 = min(y21, y22)
        xi2 = min(x21, x22)
        inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
        #  Union(A,B) = A + B - Inter(A,B)
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x22 - x12) * (y22 - y12)
        union_area = box1_area + box2_area - inter_area
        # Calcul de IOU
        iou = inter_area / union_area

        if iou > ovthresh:
            tp[d] = 1.0
        else:            
            fp[d] = 1.0
        d += 1
        
    
    # Calcul de la précision et du recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    
    return prec, rec


def compute_ap_and_recall(all_bdbox, all_gt, ovthresh):
    """
        Calcul de la VOC detection metrique. 
    """
    prec, rec = prec_rec_compute(all_bdbox, all_gt, ovthresh)
    ap = voc_ap(rec, prec, False)
    return ap, rec[-1]


def eval_stats_at_threshold( all_bdbox, all_gt, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
        Evaluation et collecte des statistiques et ce pour différents seuils.
    """
    stats = {}
    for ovthresh in thresholds:
        ap, recall = compute_ap_and_recall(all_bdbox, all_gt, ovthresh)
        stats[ovthresh] = {'ap': ap, 'recall': recall}
    stats_df = pd.DataFrame.from_records(stats)*100
    return stats_df


"""
    Structure de données pour stocker les éléments de mémoire pour l'algorithme de Replay Memory.
"""
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
