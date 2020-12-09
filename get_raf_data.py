# get Data
import os
import torchvision as tv
import torch
from generator_RAF import RAFDataset

def get_dataloaders(*p_data):
    
    #mean size (360, 475), ou (100, 132)
    size=p_data[-1]
    batch_size=p_data[0]
    data_augmentation=p_data[-1]
    params_train=[data_augmentation]
    
    # what transformations should be done with our images
    data_transforms = tv.transforms.Compose([
        tv.transforms.Resize(size),
        tv.transforms.ToTensor(),
    ])
    
    # initialize our dataset at first
    train_dataset = RAFDataset(
        d_set='train',
        params=params_train,
        transform=data_transforms
    )
    validation_dataset = RAFDataset(
        d_set='validation',
        params=params_train,
        transform=data_transforms
    )

    test_dataset = RAFDataset(
        d_set='test',
        params=params_train,
        transform=data_transforms
    )
    # initialize data loader with required number of workers and other params
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=16)

    validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=16)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=16)
    
    torch.save(train_loader,'DataLoader/Rafdb_train_set_'+str(batch_size)+'_'\
               +str(size)+'.pth')
    
    torch.save(validation_loader,'DataLoader/Rafdb_valid_set_'+str(batch_size)+'_'\
               +str(size)+'.pth')
    torch.save(test_loader,'DataLoader/Rafdb_test_set_'+str(batch_size)+'_'\
               +str(size)+'.pth')
    return train_loader,validation_loader,test_loader


def load_data(batch_size=16,size=(152,120),data_augmentation=False):
    
    
    p_data=[batch_size,data_augmentation,size]
    
    try:
        
        trainloader=torch.load('DataLoader/Rafdb_train_set_'+str(batch_size)+'_'\
                               +str(size)+'.pth')
        validloader=torch.load('DataLoader/Rafdb_valid_set_'+str(batch_size)+'_'\
                               +str(size)+'.pth')
        testloader=torch.load('DataLoader/Rafdb_test_set_'+str(batch_size)+'_'\
                               +str(size)+'.pth')
        
    except:
        trainloader,validloader,testloader=get_dataloaders(*p_data)
        
    print('num batch train:',len(trainloader))
    print('num batch validation:',len(validloader))
    print('num batch test:',len(testloader))
    
    return trainloader,validloader,testloader


