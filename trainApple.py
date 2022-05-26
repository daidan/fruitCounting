'''
Created on Nov 7, 2020

@author: daidan
'''
import copy
import json
import os
import random
import sys
import time
import warnings
import math
import h5py
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch.autograd import Variable
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import lr_scheduler
from torchvision import transforms

from CANNet import CANNet
import PIL.Image as Image
import dataset
import numpy as np
import torch.nn as nn
from utils import save_checkpoint

workers=5
epochs=300
momentum = 0.95
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def get_current_lr(optimizer, group_idx, parameter_idx):
    # Adam has different learning rates for each paramter. So we need to pick the
    # group and paramter first.
    group = optimizer.param_groups[group_idx]
    p = group['params'][parameter_idx]

    #beta1, _ = group['betas']
    beta1,beta2= group['betas']
    state = optimizer.state[p]
    t=state['step']

    bias_correction1 = 1 - beta1 ** state['step']
    current_lr = group['lr'] / bias_correction1

    #current_lr = group['lr'] * math.sqrt(1 - beta2**t) / (1 - beta1**t)
    return current_lr


def main(batch_size,lr,modelFile,loadModel):

    decay=5*1e-4

    preTrain_json="./datasets/acfr_apples/apples_train_.json"

    preVal_json="./datasets/acfr_apples/apples_val_.json"

    preTest_json="./datasets/acfr_apples/apples_test_.json"
    

    with open(preTrain_json, 'r') as outfile:
        preTrain_list = json.load(outfile)
    print(len(preTrain_list))

    with open(preVal_json, 'r') as outfile:
        preVal_list = json.load(outfile)

#     preVal_list=random.sample(preVal_list, 16)
    print(len(preVal_list))

    with open(preTest_json, 'r') as outfile:
        preTest_list = json.load(outfile)
    print(len(preTest_list))



    model=CANNet().to(device)

    criterion = nn.MSELoss(size_average=False).to(device)

    #optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=decay, momentum=0.9)

    
    # optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=decay)
    
    checkpoint = torch.load(loadModel)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=decay)

    # ma=0.1
#     scheduler=lr_scheduler.MultiStepLR(optimizer, milestones=[23], gamma=0.1)
#for Adam cycle_momentum=False
    scheduler=lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=1e-3, step_size_up=50,cycle_momentum=False)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
    modelTL,maeTrain,lossTrain,maeVal,maeTest,rmseTest,currLr=train(preTrain_list,preVal_list,preTest_list, model,
                                                 criterion, optimizer, scheduler, trainC=True,
                                                 batchSize=batch_size,modelFile=modelFile)


    print('save the loss, mae .....')

    savePara(modelFile,maeTrain,lossTrain,maeVal,maeTest,rmseTest,currLr)

def savePara(file,maeTrain,lossTrain,maeVal,maeTest,rmseTest,currLr):

    maeFile=open(file+'maeTrain.txt','w')
    maeFile.write(str(maeTrain))

    lossFile=open(file+'lossTrain.txt','w')
    lossFile.write(str(lossTrain))

    valFile=open(file+'maeVal.txt','w')
    valFile.write(str(maeVal))

    testmaeFile=open(file+'maeTest.txt','w')
    testmaeFile.write(str(maeTest))

    rmseFile=open(file+'rmseTest.txt','w')
    rmseFile.write(str(rmseTest))

    currLrFile=open(file+'currLr.txt','w')
    currLrFile.write(str(currLr))

def train(train_list,preVal_list,preTest_list, model, criterion, optimizer, scheduler, trainC, batchSize, modelFile):
    best_mae=5.0
    currLr=[]
    maeTrain=[]
    lossTrain=[]
    maeVal=[]
    maeTest=[]
    rmseTest=[]

    train_loader = torch.utils.data.DataLoader(
            dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                    ]),
                       train=trainC,
                       seen=model.seen,
                       batch_size=batchSize,
                       num_workers=workers),
        batch_size=batchSize)

    since = time.time()

    for epoch in range(0, epochs):


        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        scheduler.step()
        
        Initial_lr=optimizer.param_groups[0]['initial_lr']
        CurrLr=optimizer.param_groups[0]['lr']

        print('epoch %d, processed %d samples,initial lr %.10f,curr lr %.10f' % (epoch, epoch * len(train_loader.dataset),Initial_lr,CurrLr))
        
        model.train()
        mae=0
        running_loss=0
        end = time.time()

        for i,(img, target)in enumerate(train_loader):

            data_time.update(time.time() - end)

            img = img.to(device)
            img = Variable(img)
           # with torch.no_grad():
            output = model(img)[:,0,:,:]

            target = target.type(torch.FloatTensor).to(device)
            target = Variable(target)

            loss = criterion(output, target)

            losses.update(loss.item(), img.size(0))
            running_loss+=loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            
            currLr.append(optimizer.param_groups[0]['lr'])

            batch_time.update(time.time() - end)
            end = time.time()

            mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor)).cpu().numpy()

            if i % 5 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'maeBatch {maeBatch:.3f}\t'
                      .format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, maeBatch=mae))

        print('learning rate:',scheduler.get_last_lr())
        group_idx, param_idx = 0, 0
        current_lr = get_current_lr(optimizer, group_idx, param_idx)
        currLr.append(current_lr)
        print('Current learning rate (g:%d, p:%d): %.7f'%(group_idx, param_idx, current_lr))
        
        modelEpoch=[67,117,167,200,217,267]
        if epoch in modelEpoch:
            print('save the model in epoch:',epoch)
            save_checkpoint({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'best_prec1': best_mae,
                        'optimizer_state_dict' : optimizer.state_dict(),
                    }, best_mae, modelFile, filename='Model_'+str(epoch)+'.pth.tar')

        mae = mae/len(train_loader)
        print(' * MAE {mae:.3f} '.format(mae=mae))
        maeTrain.append(mae)

        vaule=running_loss/len(train_loader)
        lossTrain.append(vaule)
        print(' * loss {lossValue:.3f} '.format(lossValue=vaule))

        prec1 = validate(preVal_list, model)
        maeVal.append(prec1)

        print(' * Val MAE {mae:.3f} '.format(mae=prec1))
        if prec1 < best_mae:
            best_mae = prec1

            print(' * best MAE {mae:.3f} '.format(mae=best_mae))
            save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_prec1': best_mae,
                    'optimizer_state_dict' : optimizer.state_dict(),
                }, best_mae, modelFile, filename='checkpoint.pth.tar')


        print('the test')

        maeV,rmseV=test(preTest_list,model)

        maeTest.append(maeV)
        rmseTest.append(rmseV)

        print(' * Test MAE {mae:.3f} '.format(mae=maeV))

    time_elapsed = time.time() - since

    data_time.update(time.time() - since)

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model,maeTrain,lossTrain,maeVal,maeTest,rmseTest,currLr


def validate(val_list, model):
    print ('begin val')
    val_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=1)

    model.eval()

    mae = 0

    for i,(img, target) in enumerate(val_loader):
        h,w = img.shape[2:4]
        h_d = int(h/2)
        w_d = int(w/2)
        img_1 = Variable(img[:,:,:h_d,:w_d].cuda())
        img_2 = Variable(img[:,:,:h_d,w_d:].cuda())
        img_3 = Variable(img[:,:,h_d:,:w_d].cuda())
        img_4 = Variable(img[:,:,h_d:,w_d:].cuda())
        density_1 = model(img_1).data.cpu().numpy()
        density_2 = model(img_2).data.cpu().numpy()
        density_3 = model(img_3).data.cpu().numpy()
        density_4 = model(img_4).data.cpu().numpy()

        pred_sum = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()

        mae += abs(pred_sum-target.sum()).cpu().numpy()

    mae = mae/len(val_loader)
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return mae


def test(img_paths, model):

    pred= []
    gt = []

    transform=transforms.Compose([
        transforms.ToTensor(),transforms.Normalize(mean=[0.48,0.456,0.406],
            std=[0.229,0.224,0.225]),
        ])

    for i in range(len(img_paths)):
        img = transform(Image.open(img_paths[i]).convert('RGB')).to(device)
        img = img.unsqueeze(0)
        h,w = img.shape[2:4]
        h_d = int(h/2)
        w_d = int(w/2)
        img_1 = Variable(img[:,:,:h_d,:w_d].to(device))
        img_2 = Variable(img[:,:,:h_d,w_d:].to(device))
        img_3 = Variable(img[:,:,h_d:,:w_d].to(device))
        img_4 = Variable(img[:,:,h_d:,w_d:].to(device))
        density_1 = model(img_1).data.cpu().numpy()
        density_2 = model(img_2).data.cpu().numpy()
        density_3 = model(img_3).data.cpu().numpy()
        density_4 = model(img_4).data.cpu().numpy()

        if 'jpg' in img_paths[i]:
            imgType='.jpg'
        else:
            imgType='.png'

        pure_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
        gt_file = h5py.File(img_paths[i].replace(imgType,'.h5'),'r')
        groundtruth = np.asarray(gt_file['density'])
        pred_sum = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()
        pred.append(pred_sum)
        gt.append(np.sum(groundtruth))

    mae = mean_absolute_error(pred,gt)
    rmse = np.sqrt(mean_squared_error(pred,gt))

    print('MAE: ',mae)
    print('RMSE: ',rmse)

    return mae, rmse

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def batchSize():
    loadModel='./modelPara/almond/batchSize32/best_checkpoint.pth.tar'
    saveModelBase='./modelPara/apple/'
    batchSizeList=[4,5,8,10]
    learningRateList={"1e-4":1e-4,"1e-5":1e-5,"1e-6":1e-6}
    for batch_size in batchSizeList:
        for filename,lr in learningRateList.items():
            print('this training batch size:',batch_size)
            print('this training learning rate:',lr)
             
            saveModelFile=saveModelBase+'batchSize'+str(batch_size)+'/'+filename+'/'
            print('the path of model:',saveModelFile)
             
            if not os.path.exists(saveModelFile):
                os.mkdir(saveModelFile)
             
            main(batch_size=batch_size,lr=lr,modelFile=saveModelFile,loadModel='')



def basicModel():
    
    batch_size=4
    learningRateList={"1e-4":1e-4,"1e-5":1e-5}
     
    saveModelBase='./modelPara/apple/Cosine/batch4/'
     
    for filename,lr in learningRateList.items():
         
        saveModelFile=saveModelBase+filename+'/'
        print('the path of model:',saveModelFile)
         
        if not os.path.exists(saveModelFile):
            os.mkdir(saveModelFile)
     
        main(batch_size=batch_size,lr=lr,modelFile=saveModelFile,loadModel='')
        

def lrStepDecay():
    batch_size=4
    
    modelEpoch={"Model_67.pth.tar":67,"Model_117.pth.tar":117,"Model_167.pth.tar":167,"Model_217.pth.tar":217,
                "Model_267.pth.tar":267,"Model_317.pth.tar":317,"Model_367.pth.tar":367,"Model_417.pth.tar":417,
                "Model_467.pth.tar":467,"Model_517.pth.tar":517,"Model_567.pth.tar":567}

    
    lrList=[1e-5,1e-6]
    
    for lr in lrList:
        
        if lr==1e-5:
    
            saveModelBase='./modelPara/apple/Adam/batch4/1e-4/'
            
        if lr ==1e-6:
            
            saveModelBase='./modelPara/apple/Adam/batch4/1e-5/'
        
        for modelName,epoch in modelEpoch.items():
            
            loadModelFile=saveModelBase+modelName
            
            saveFile=saveModelBase+str(epoch)+'/'
            
            print('lr',lr)
            print('the path of save:',saveFile)
             
            if not os.path.exists(saveFile):
                os.mkdir(saveFile)
                
            main(batch_size=batch_size,lr=lr,modelFile=saveFile,loadModel=loadModelFile)


def cyclicalModel():
    
    saveModelBase='./modelPara/apple/SGD/cyclicalModel/'
    batchSizeList=[1,4,8,16,32]
    
    lr=1e-7
    
    for batch_size in batchSizeList:
        
        saveModelFile=saveModelBase+'batchSize'+str(batch_size)
        
        print('the path of model:',saveModelFile)
             
        if not os.path.exists(saveModelFile):
            os.mkdir(saveModelFile)
         
        main(batch_size=batch_size,lr=lr, modelFile=saveModelFile,loadModel='')


if __name__ == '__main__':
    
    cyclicalModel()
    
    

        
