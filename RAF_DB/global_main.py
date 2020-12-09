
from get_raf_data import *
from get_model import load_model
from train_model_raf import train_model
from torch.utils.tensorboard import SummaryWriter
#hyperparameters tuning
""" model / layer / batch_size / learning_rate / data_augmentation / dropout_prob / criterion (ArcFaceLoss) / """ 


def main(**kwargs):
    
    ######   Create parameters variables   ######
    print('training parameters: ')
    for k, v in kwargs.items():
        globals()[k]=v
        print(k, '=', v)
    
    
    ######   Create dictionary of variables.  ######
    units=dict()
    #initialize global variables
    size=(160,120)
    
    cp_filename=os.path.join('./Checkpoints/',filename+'.pth.tar')
    
    csv_path=os.path.join('./Logs/',filename+'.csv')
    
    if os.path.exists(cp_filename):
        print("This model already exists")
        return True
  
    trainloader,validloader,testloader=load_data(batch_size=batch_size,
                                                     size=size,
                                                     data_augmentation=False)
    
    model,optimizer,criterion,device=load_model(baseline=model_name,
                                            freeze_layer=layer, dropout_prob=dropout_prob,
                                            GPU=0,lr=lr)
    
    comment = f' layer= {layer} batch_size = {batch_size} lr = {lr} dropout_prob = {dropout_prob} data_augmentation= {data_augmentation}' 
    
    f=os.path.join('./runs',filename)
    tb = SummaryWriter(log_dir=f,comment=comment)
    
    params=[layer,batch_size,lr,dropout_prob,data_augmentation]
    
    train_model(tb,model,optimizer,criterion,
          trainloader,validloader, cp_filename, csv_path,
          device=device,params=params)

    tb.close()

    return True

if __name__ == '__main__':
    main()
        