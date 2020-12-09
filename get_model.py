#import parent folder
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir))

import numpy as np
from Models.inception_resnet_v1 import InceptionResnetV1 
import torch
import torch.nn as nn
from torchsummary import summary
from vgg_face_pytorch_master.models.vgg_face import VGG_16
from vgg_face_pytorch_master.models.vgg_face_bn import VGG_16_bn
from Models.resnet50_ft_dag import Resnet50_ft_dag
from facenet_pytorch_master.models.mtcnn import MTCNN
from Models.inception_resnet_v1 import InceptionResnetV1
from sklearn.utils.class_weight import compute_class_weight
import pkbar
import csv

def get_class_weights():
    annotations=open('../Dataset/RAF Dataset/Basic Emotion/EmoLabel/list_patition_label.txt','r').readlines()
    labels=[int(elt.split(' ')[1].replace('\n','')) for elt in annotations]
    
    train_labels=np.asarray(labels[int(0.1*12270):])
    class_weights=compute_class_weight('balanced',np.arange(1,8,1), train_labels)
    return class_weights

def load_model(baseline='inception_resnetV1',freeze_layer='conv_5_1', dropout_prob=0.6,
          GPU=0,lr=1e-5,show_model=False):
    
    #torch.cuda.set_device(GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if baseline=='inception_resnetV1':
        units=dict()
        units['classify']=True
        units['regression']=False
        units['multiplier']=1
        units['num_classes']=7
        units['fc6_in'],units['fc6_out'] = 512,128
        units['fc7_in'] = 128
        model=InceptionResnetV1(pretrained="vggface2", dropout_prob=dropout_prob, device=None, **units)
        
    
    layers_name=[]
    for name, child in model.named_children():
        layers_name.append(name)
    
    #conv2d_2a / conv2d_2b / conv2d_3b/ conv2d_4a/ conv2d_4b/ repeat_1/ mixed_6a/ repeat_2/ mixed_7a/ repeat_3/ block8/ last_linear  

    if freeze_layer == None :
        for param in model.parameters():
            param.requires_grad = True
    else:
        list_layers=[]
        idx=layers_name.index(freeze_layer)
        list_layers=layers_name[idx:]
        unfreeze=False
        for name, child in model.named_children():
            
            if name in list_layers:
                unfreeze=True
                #print(name + ' is unfrozen')
                for param in child.parameters():
                    param.requires_grad = unfreeze
            else:
                #print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = unfreeze

    class_weights=get_class_weights()
    class_weights=torch.Tensor(class_weights).to(device)
    
    criterion = nn.NLLLoss(weight=class_weights)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=lr,amsgrad=False)
    model.to(device)
    if show_model:
        summary(model,input_size=(3,160,120))
    #summary(model,input_size=(3,100,80))
    
    return model,optimizer,criterion,device