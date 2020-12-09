
from audtorch.functional import *
from audtorch.metrics import *
from callbacks import *

import pkbar
import numpy as np

import torch
from torchsummary import summary
import torch.nn as nn
import torch.optim as opt
torch.set_printoptions(linewidth=120)
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


def train_model(tb,model,optimizer,criterion,
          trainloader,validloader, filename, csv_path,
          device=0,params=[]):
    
    layer,batch_size,lr,dropout_prob,data_augmentation=params
    print('Train')
    #hyperparameters initialization
    epochs = 20
    best_loss=1000.0
    start_epoch=0
    early_stop=True
    counter_early_stop=0
    patience=5
    
    #Train loop
    
    for epoch in range(epochs):
        
        model.train()
        train_temp_accuracy=0
        print('Epoch: %d/%d' % (epoch + 1, epochs))
        kbar = pkbar.Kbar(target=len(trainloader), width=50)
        running_loss = 0
        steps = 0

        for batch_id, (inputs, labels) in enumerate(trainloader):
            #labels-=1
            steps += 1
            labels = labels.squeeze(1)
            labels=torch.argmax(labels,axis=1)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loss=running_loss/steps
            
            ps= torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_temp_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_accuracy= train_temp_accuracy/steps
            
            kbar.update(steps, values=[("loss", train_loss), ("acc", train_accuracy)])  
            
            tb.add_scalar("Batch_Loss",loss, batch_id+((epoch*len(trainloader))))
            
        tb.add_scalar("Train_Loss", train_loss, epoch)
        tb.add_scalar("Train_Accuracy", train_accuracy, epoch)
        
        


        """for name, weight in model.named_parameters():
            tb.add_histogram(name,weight, epoch)
            tb.add_histogram(f'{name}.grad',weight.grad, epoch)"""
        
        steps=0
        val_temp_loss=0
        val_temp_accuracy=0

        model.eval()
        with torch.no_grad():
            for inputs, labels in validloader:
                
                steps+=1
                labels = labels.squeeze(1)
                labels=torch.argmax(labels,axis=1)
                inputs, labels = inputs.to(device),labels.to(device)
                
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                val_temp_loss += batch_loss.item()
                
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)  
                equals = top_class == labels.view(*top_class.shape)
                val_temp_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            val_accuracy = val_temp_accuracy/steps
            val_loss=val_temp_loss/steps
            
            tb.add_scalar("Val_Loss", val_loss, epoch)
            tb.add_scalar("Val_Accuracy", val_accuracy, epoch)
            
            # Callbacks
            is_best = bool(val_loss < best_loss)
            # Get greater Tensor to keep track best acc
            best_loss = min(val_loss, best_loss)
            
            # Save checkpoint if is a new best
            ModelCheckpoint({
                'epoch': start_epoch + epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss
            }, is_best, filename)

            # EarlyStopper
            early_stop,counter_early_stop=EarlyStopper(patience, is_best, counter_early_stop)
            if early_stop==True:
                break

            
            kbar.add(1, values=[("loss", train_loss), ("acc", train_accuracy),
                                ("val_loss", val_loss), ("val_acc", val_accuracy)])
            tb.add_hparams(
                {"lr": lr, "bsize": batch_size, "layer":layer, "dropout_prob":dropout_prob,
                 "data_augmentation":data_augmentation},
            {
                "train_accuracy": train_accuracy,
                "train_loss": train_loss,
                "val_accuracy": val_accuracy,
                "val_loss": val_loss,
            },
        )
    
    return model