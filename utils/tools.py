
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

classes = ['cat', 'bird', 'motorbike', 'diningtable', 'train', 'tvmonitor', 'bus', 'horse', 'car', 'pottedplant', 'person', 'chair', 'boat', 'bottle', 'bicycle', 'dog', 'aeroplane', 'cow', 'sheep', 'sofa']

def sort_class_extract(datasets):    
    datasets_per_class = {}
    for j in classes:
        datasets_per_class[j] = []

    for dataset in datasets:
        for i in dataset:
            img, target = i
            classe = target['annotation']['object'][0]["name"]
            datasets_per_class[classe].append(i)
           
    return datasets_per_class


def show_new_bdbox(image, labels, color='r'):
    xmin, xmax, ymin, ymax = labels[0],labels[1],labels[2],labels[3]
    fig,ax = plt.subplots(1)
    ax.imshow(image.transpose(0, 2).transpose(0, 1))

    width = xmax-xmin
    height = ymax-ymin
    rect = patches.Rectangle((xmin,ymin),width,height,linewidth=3,edgecolor=color,facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()

def show_bdbox(train_loader, index):
    fig,ax = plt.subplots(1)
    img, target = train_loader[index]
    ax.imshow(img.transpose(0, 2).transpose(0, 1))
    #print("Labels : "+str(labels['annotation']['size']))
    
    print(img)
    print(target)
    xmin = ( int(target['annotation']['object'][0]['bndbox']['xmin']) /  int(target['annotation']['size']['width']) ) * 224
    xmax = ( int(target['annotation']['object'][0]['bndbox']['xmax']) /  int(target['annotation']['size']['width']) ) * 224

    ymin = ( int(target['annotation']['object'][0]['bndbox']['ymin']) /  int(target['annotation']['size']['height']) ) * 224
    ymax = ( int(target['annotation']['object'][0]['bndbox']['ymax']) /  int(target['annotation']['size']['height']) ) * 224
    
    width = xmax-xmin
    height = ymax-ymin
    rect = patches.Rectangle((xmin,ymin),width,height,linewidth=3,edgecolor='r',facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()

def extract(index, loader):
    img, target = loader[index]
    xmin = ( int(target['annotation']['object'][0]['bndbox']['xmin']) /  int(target['annotation']['size']['width']) ) * 224
    xmax = ( int(target['annotation']['object'][0]['bndbox']['xmax']) /  int(target['annotation']['size']['width']) ) * 224

    ymin = ( int(target['annotation']['object'][0]['bndbox']['ymin']) /  int(target['annotation']['size']['height']) ) * 224
    ymax = ( int(target['annotation']['object'][0]['bndbox']['ymax']) /  int(target['annotation']['size']['height']) ) * 224

    width = xmax-xmin
    height = ymax-ymin
    
    return img, [xmin, xmax, ymin, ymax]
