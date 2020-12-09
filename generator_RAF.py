#create Dataloaders

import os
import numpy as np
import csv
import torch
import torchvision as tv
import random
import PIL
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

class RAFDataset(torch.utils.data.Dataset):
    def __init__(self, d_set=None, params=None, transform=None,
                 loader=tv.datasets.folder.default_loader):
        
        assert params is not None and d_set is not None
        
        self.transform = transform
        self.loader = loader
        self.d_set=d_set
        self.data_augmentation=params[0]
        
        self.data,self.targets=self.get_data()
        
    def get_data(self):
        path="../Dataset/RAF Dataset/Basic Emotion/Image/aligned"
        
        annotations=open('../Dataset/RAF Dataset/Basic Emotion/EmoLabel/list_patition_label.txt','r').readlines()
        img_names=[elt.split(' ')[0].replace('.jpg','_aligned.jpg') for elt in annotations]
        img_names=[os.path.join(path,img) for img in img_names]
        
        labels=[int(elt.split(' ')[1].replace('\n','')) for elt in annotations]
        
        if self.d_set=='train':
            return img_names[int(0.1*12270):12270],labels[int(0.1*12270):12270]
        elif self.d_set=='validation':
            return img_names[:int(0.1*12270)],labels[:int(0.1*12270)]
        elif self.d_set=='test':
            return img_names[12271:],labels[12271:]
       
    
    def __getitem__(self, index):
        
        frame_path=self.data[index]
        target=self.targets[index]
        target=torch.from_numpy(np.asarray(target))
        target = target.reshape(1, 1)
        num_classes = 7
        target = (target == torch.arange(num_classes)).long()
        
        if self.data_augmentation:
            p = 0.5
            r1=random.uniform(0,1)
            r2=random.uniform(0,1)
            r3=random.uniform(0,1)
            r4=random.uniform(0,1)
            v1=random.uniform(1,1.5)
            v2=random.uniform(1,1.5)
            h_shift=random.randint(-10,10)
            v_shift=random.randint(-10,10)
                
        img = self.loader(frame_path)
        
        if self.data_augmentation:
                #img=tv.transforms.functional.to_pil_image(img, mode='RGB')
                if 0<=r1<p:
                    img=tv.transforms.functional.hflip(img)
                if 0<=r2<p:   
                    img=tv.transforms.functional.adjust_brightness(img, v1)
                if 0<=r3<p:
                    img=tv.transforms.functional.adjust_contrast(img, v2)
                if 0<=r4<p:    
                    img=tv.transforms.functional.affine(img, angle=0, translate=(h_shift,v_shift),
                                                        scale=1.,
                                                        shear=0,resample=PIL.Image.NEAREST, 
                                                        fillcolor=None)  
        if self.transform is not None:
                img = self.transform(img)
                img = img.numpy()
        
        frame=torch.from_numpy(img)
        
        return frame, target

    
    def __len__(self):
        n = len(self.data)
        return n
    