import os
from glob import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
tqdm.pandas()
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error
import timm
from accelerate import Accelerator
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import gc
from copy import deepcopy


# In[17]:


parameters = {
    'nb_epochs': 100,
    'nsc' : 30,
    'base_model': "convnext_small.fb_in22k_ft_in1k",
    'checkpoint': "./checkpoint_cv_ct",
    'batch_size': 64,
    'image_size' : 256,
    'weight_decay': 1e-9,
    'seed': 0,
    'lr': 2e-4,
    'gamma': 0.32,
    'nsplits_test': 5,
    'nsplits': 4,
}


# In[18]:


def path2info(row):
    path = row['image_path']
    data = path.split('/')[-1]
    image = data.split('.')[0]
    row['image'] = image
    return row


# In[19]:


file_lbl="data/Retinopathy/trainLabels.csv"
paths= glob('data/Retinopathy/train/*.jpeg')

path_df = pd.DataFrame(paths, columns=['image_path'])
path_df = path_df.progress_apply(path2info, axis=1)
df=pd.read_csv(file_lbl,sep=',')
df_ret = df.merge(path_df, on=['image'])
df, final_df_test =  train_test_split(df_ret,  test_size=0.9,
                                      stratify=df_ret['level'],
                                      random_state=parameters['seed'])


# In[23]:


class RetDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.image_path = df['image_path'].tolist()
        self.lbl = df['level'].tolist()
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lbl=torch.tensor([self.lbl[idx]], dtype=torch.float32)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, lbl


# In[24]:


image_size = parameters['image_size']
batch_size = parameters['batch_size']
transform_train = A.Compose([
    A.Resize(image_size,image_size),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Rotate(90),
    A.HueSaturationValue(),
    A.RandomGamma(),
    A.RandomBrightnessContrast(),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
    ])

transform_test = A.Compose([
    A.Resize(image_size,image_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
    ])


# In[25]:


def train(epoch,accelerator, trainloader,net, optimizer):
    net.train()
    train_loss = 0
    total = 0
    correct=0
    criterion = nn.MSELoss()
    loop = tqdm(enumerate(trainloader), total=len(trainloader), 
                disable=not accelerator.is_local_main_process)
    for batch_idx, (inputs, targets) in loop:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
        train_loss += loss.item()*targets.size(0)
        predicted = torch.clamp(torch.round(outputs), min=0, max=4)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        loop.set_description(f"Epoch [{epoch}]")
        loop.set_postfix(loss=train_loss/total,acc=correct/total)
    return train_loss/total, correct/total


# In[26]:


def validation(accelerator,validloader, net, best_acc):
    device=accelerator.device
    is_best=False
    net.eval()
    valid_loss = 0
    total = 0
    correct=0
    criterion = nn.MSELoss()
    predicted_scores = torch.empty(0)
    vrais_label = torch.empty(0)
    with torch.no_grad():
        loop = tqdm(enumerate(validloader), total=len(validloader), 
                    disable=not accelerator.is_local_main_process)
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            valid_loss += loss.item() * targets.size(0)
            predicted = torch.clamp(torch.round(outputs), min=0, max=4)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            predicted_scores=torch.cat((predicted_scores, 
                                            torch.reshape(outputs, (-1,)).to('cpu')))
            vrais_label = torch.cat((vrais_label, 
                                         torch.reshape(targets, (-1,)).to('cpu')))
            loop.set_postfix(loss=valid_loss/total, acc=correct/total)
    acc = valid_loss/total
    if  acc < best_acc:
        is_best=True
    return acc, is_best


# In[27]:


def predict(accelerator, testloader, net):
    device=accelerator.device
    net.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    total = 0
    correct = 0
    predicted_outputs = torch.empty(0)
    with torch.no_grad():
        loop = tqdm(enumerate(testloader), total=len(testloader), disable=not accelerator.is_local_main_process)
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs=net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            predicted = torch.clamp(torch.round(outputs), min=0, max=4)
            correct += predicted.eq(targets).sum().item()
            predicted_outputs = torch.cat((predicted_outputs,outputs.to('cpu')))
            loop.set_postfix(loss=test_loss/total, acc=correct/total)
    return  predicted_outputs


# In[28]:


def evaluation(accelerator,net, 
               finalevaluation_dataloader, 
               len_finalevaluation_data,
               root_namesave = "./checkpoint"):
        net=net.to(accelerator.device)
        prediction_outputs=torch.zeros(len_finalevaluation_data)
        for fold_test in range(parameters['nsplits_test']):
            for fold in range(parameters['nsplits']):
                namesave=root_namesave+'/ckpt_'+str(fold_test)+'_'+str(fold)+".pt"
                checkpoint=torch.load(namesave, map_location=accelerator.device)
                net.load_state_dict(checkpoint['net'])
                pred_outputs = predict(accelerator, finalevaluation_dataloader, net)
                prediction_outputs += pred_outputs.squeeze(1)/(parameters['nsplits_test']*parameters['nsplits'])
        return prediction_outputs.numpy()
        


def train_onefold(accelerator, net, df, fold_valid, fold_test, train_idx, valid_idx, root_namesave):
    lr = parameters['lr']
    ne = parameters['nb_epochs']
    nsc = parameters['nsc']
    gamma = parameters['gamma']
    wd=parameters['weight_decay']
    df_train=df.iloc[train_idx].copy()
    df_train.reset_index(drop=True, inplace=True)
    df_valid=df.iloc[valid_idx].copy()
    df_valid.reset_index(drop=True, inplace=True)
    training_data=RetDataset(df_train,transform=transform_train)
    valid_data=RetDataset(df_valid,transform=transform_test)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, 
                              shuffle=True, num_workers=2, pin_memory=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, 
                              shuffle=False, num_workers=2, pin_memory=True)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lr_sc = lr_scheduler.StepLR(optimizer, step_size=nsc, gamma=gamma)
    net, optimizer, train_dataloader, lr_sc = accelerator.prepare(net, 
                                                                  optimizer, 
                                                                  train_dataloader, 
                                                                  lr_sc)
    best_acc=1e9
    namesave=root_namesave+f"/ckpt_{fold_test}_{fold_valid}.pt"
    for epoch in range(0, ne):
        train(epoch, accelerator, train_dataloader,net, optimizer)
        accelerator.wait_for_everyone()
        lr_sc.step()
        if accelerator.is_main_process:
            valid_acc, is_best = validation(accelerator, valid_dataloader, 
                                            net, best_acc)
            print("MSE valid : ",valid_acc)
            if is_best:
                best_acc=valid_acc
                print("Saving...")
                unwrapped_net = accelerator.unwrap_model(net)
                state = {
                    'net': unwrapped_net.state_dict(),
                    'acc': valid_acc,}
                torch.save(state,namesave)
    return best_acc

def train_kfold(accelerator, base_model, df_app, fold_test, root_namesave):
    ACC_=np.zeros(parameters['nsplits'])
    gsfk=StratifiedKFold(parameters['nsplits'], shuffle=True, random_state=parameters['seed'])
    splits = list(gsfk.split(df_app,df_app['level']))
    for fold, (train_idx, valid_idx) in enumerate(splits):
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.free_memory()
        net=deepcopy(base_model)
        ACC_[fold] = train_onefold(accelerator, net, df_app, fold, fold_test, train_idx, valid_idx, root_namesave)
    accelerator.wait_for_everyone()
    return ACC_

def final_test(accelerator, base_model, df_test, fold_test, root_namesave):
        net=deepcopy(base_model).to(accelerator.device)
        test_data=RetDataset(df_test, transform = transform_test)
        len_test_data=len(test_data)
        prediction_outputs=torch.zeros(len_test_data)
        test_dataloader = DataLoader(test_data, batch_size=parameters['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
        for fold_valid in range(parameters['nsplits']):
            namesave=parameters['checkpoint']+'/ckpt_'+str(fold_test)+"_"+str(fold_valid)+".pt"
            checkpoint=torch.load(namesave, map_location=accelerator.device)
            net.load_state_dict(checkpoint['net'])
            pred_outputs = predict(accelerator, test_dataloader,net)
            prediction_outputs += pred_outputs.squeeze(1)/parameters['nsplits']
        mse_test=mean_squared_error(df_test['level'],prediction_outputs.numpy())
        return mse_test




def main():
    # definition of hypererparameters
    accelerator = Accelerator(mixed_precision="fp16")
    root_namesave = parameters['checkpoint']
    base_model=timm.create_model(parameters['base_model'], 
                                 pretrained=True, num_classes=1)
    if accelerator.is_main_process:
        if not os.path.isdir(root_namesave):
            os.mkdir(root_namesave)
    ACC_valid=np.zeros((parameters['nsplits_test'],parameters['nsplits']))
    ACC_test=np.zeros(parameters['nsplits_test'])
    gskf_test = StratifiedKFold(parameters['nsplits_test'], 
                                shuffle=True, random_state=parameters['seed'])
    splits_test = list(gskf_test.split(df,df['level']))
    for fold_test, (train_idx, test_idx) in enumerate(splits_test):
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.free_memory()
        df_app=df.iloc[train_idx].copy()
        df_app.reset_index(drop=True, inplace=True)
        df_test=df.iloc[test_idx].copy()
        df_test.reset_index(drop=True, inplace=True)
        net = deepcopy(base_model)
        ACC_valid[fold_test]=train_kfold(accelerator, net, df_app, fold_test, root_namesave)
        if accelerator.is_main_process:
            print('fold_test: ',fold_test)
            gc.collect()
            torch.cuda.empty_cache()
            accelerator.free_memory()
            ACC_test[fold_test]=final_test(accelerator, base_model, df_test,fold_test, root_namesave)              
            print("Validation MSE  : ", ACC_valid[fold_test])
            mean_acc_moy_valid= np.mean(ACC_valid[fold_test])
            print("Mean of validation MSE: ", mean_acc_moy_valid)
            print("Test MSE: ", ACC_test[fold_test])
    if accelerator.is_main_process:
        final_test_set = RetDataset(final_df_test,transform=transform_test)
        len_finalevaluation_data = len(final_test_set)
        finalevaluation_dataloader = DataLoader(final_test_set, 
                                                batch_size=4*parameters['batch_size'],
                                                shuffle=False, 
                                                num_workers=2, pin_memory=True)
        net = deepcopy(base_model)
        prediction_outputs = evaluation(accelerator,net,
                                        finalevaluation_dataloader, 
                                        len_finalevaluation_data,
                                        root_namesave=parameters['checkpoint'])
        labels_evaluation = np.array(final_df_test['level'])
        mean_test=np.mean(ACC_test)
        MSE_moy_evaluation = mean_squared_error(labels_evaluation,prediction_outputs)
        print('Mean MSE test ',mean_test)
        print("MSE final sur l'ensemble de test ", MSE_moy_evaluation)
        
if __name__ == "__main__":
    main()


