'''
Created on 2 Jan 2022

@author: daidan
'''
import json
import math

import matplotlib
from numpy.lib.tests.test_format import basic
from sklearn.svm import SVR

import matplotlib.pyplot as plt
import numpy as np


# totalBS=[4,2,5,5,5,4,5,8,8,8,32,64,16,32,32,32,32]
# totalSize=[[49],[86],[89],[98],[133],[172],[196],[267],[344],[392],[535],[687],[784],[1031],[1071],[1375],[2142]]
# totalBS=[4, 5, 5, 8, 16, 2, 4, 8, 64, 32, 32, 4, 5, 8, 32, 32, 32]
totalBS=[4, 4, 4, 8, 16, 2, 4, 8, 64, 32, 32, 4, 4, 8, 32, 32, 32]
totalSize=[[49],[98],[196],[392],[784],[86],[172],[344],[687],[1031],[1375],[89],[133],[267],[535],[1071],[2142]]

dataset=[[49],[98],[196],[392],[784],[86],[172],[344],[687],[1031],[1375],[89],[133],[267],[535],[1071],[2142]]
allFruitsize=[]

for i in range(0,len(dataset)):
    allFruitsize.extend([dataset[i]]*10)
    
allFruitBS=[4, 8, 2, 16, 2, 4, 2, 4, 2, 1, 4, 
            8, 8, 4, 4, 2, 4, 4, 32, 8, 2, 8, 
            4, 2, 32, 8, 32, 2, 4, 16, 64, 16, 
            2, 8, 16, 8, 8, 8, 16, 4, 8, 8, 16, 
            8, 4, 8, 16, 8, 8, 4, 4, 2, 64, 4, 2, 
            2, 2, 2, 4, 2, 2, 4, 16, 4, 8, 4, 8, 8, 
            32, 8, 64, 16, 16, 4, 4, 4, 8, 2, 8, 8, 
            32, 64, 64, 8, 64, 64, 32, 8, 32, 8, 4, 
            16, 32, 32, 64, 8, 8, 4, 32, 32, 64, 64, 
            16, 32, 32, 64, 32, 16, 64, 32, 16, 16, 4, 
            2, 1, 64, 16, 4, 1, 4, 8, 16, 4, 64, 8, 4, 4, 
            16, 4, 8, 8, 16, 32, 4, 64, 8, 32, 16, 4, 8, 
            32, 16, 8, 64, 32, 32, 8, 32, 16, 64, 32, 8, 
            64, 64, 8, 32, 32, 16, 32, 64, 16, 64, 64, 32, 
            64, 32, 32, 64, 32, 16]

# allFruitBS=[4, 8, 2, 16, 2, 4, 2, 5, 5, 1, 4, 5, 8, 4, 5, 2, 4, 4, 5, 8, 2, 8, 4, 2, 5, 8, 32, 5, 4, 16, 64, 16, 2, 8, 16, 8, 5, 5, 16, 4, 8, 8, 16, 5,
#               4, 8, 16, 8, 8, 4, 4, 5, 64, 4, 2, 2, 2, 5, 4, 2, 5, 4, 16, 4, 8, 4, 8, 8, 32, 5, 64, 16, 16, 4, 5, 4, 8, 2, 8, 8, 32, 5, 5, 5, 64, 64, 32, 
#               8, 32, 8, 4, 16, 32, 32, 64, 8, 8, 4, 32, 32, 64, 64, 16, 32, 32, 64, 32, 16, 64, 64, 16, 16, 5, 5, 1, 64, 16, 4, 1, 4, 5, 16, 4, 64, 5, 
#               5, 4, 16, 4, 8, 5, 16, 5, 4, 64, 8, 32, 16, 5, 8, 32, 16, 8, 64, 32, 32, 8, 32, 16, 64, 32, 8, 64, 64, 8, 32, 32, 16, 32, 64, 16, 64, 64, 32, 
#               64, 32, 32, 64, 32, 16]


#the average optimal batch size except bs 5
#apple [4, 5, 5, 8, 16] avg
# avgBSApple=[4, 5, 5, 8, 16] 
avgBSApple=[4, 4, 4, 8, 16] 
avgSizeApple=[[49],[98],[196],[392],[784]]

avgBSMango=[2, 4, 8, 64, 32,32]
avgSizeMango=[[86],[172],[344],[687],[1031],[1375]]

# avgBSAlmond=[4, 5, 8, 32, 32, 32]
avgBSAlmond=[4, 4, 8, 32, 32, 32]
avgSizeAlmond=[[89],[133],[267],[535],[1071],[2142]]

#total
# avgSizeApple=allFruitsize[:50]
# avgBSApple=allFruitBS[:50]
#
# avgSizeMango=allFruitsize[50:110]
# avgBSMango=allFruitBS[50:110]
#
# avgSizeAlmond=allFruitsize[110:170]
# avgBSAlmond=allFruitBS[110:170]

avgBS=[avgBSApple,avgBSMango,avgBSAlmond]
avgSize=[avgSizeApple,avgSizeMango,avgSizeAlmond]


def logRegLinearTotal():
    
    inputX=avgSize
    labelY=avgBS
    
    svr = SVR(kernel="linear", C=1, gamma="auto")
    
    fig, axes = plt.subplots()
    model_color = ["blue", "green", "black"]
    labelList=['AP', 'MA','AL']
    
    for i in range(0,3):
        
        X=inputX[i]
        Y=np.log2(labelY[i])
        
        # s=100,
        
        plt.scatter(X, Y, c=model_color[i],marker='+',s=100, label=labelList[i]+'_BS')
        plt.plot(X, svr.fit(X, Y).predict(X),
        color=model_color[i],
        lw=2,
        label=labelList[i]+"_Lin")

    plt.plot(totalSize, svr.fit(totalSize, np.log2(totalBS)).predict(totalSize),
        color='red',
        # linestyle='dashed',
        lw=2,
        label="Total_Lin")
    
    
    # plt.rcParams.update({'font.size': 12})
    plt.legend(ncol=2,loc='lower right',fontsize='14')
    plt.xlabel('Size of training data',fontsize='20')
    plt.xticks(fontsize='16')
    plt.yticks(fontsize='16')
    plt.ylabel('Log2 of batch size',fontsize='20')
    
    plt.savefig('../modelPara/batchsize/log/total170/170_linear.pdf', bbox_inches = "tight")
    # plt.savefig('../modelPara/batchsize/log/avg/17_linear.pdf', bbox_inches = "tight")


def singleLogRegLinear():
    
    X=avgSizeApple
    Y=np.log2(avgBSApple)
    
    
    # X=avgSizeMango
    # Y=np.log2(avgBSMango)
    
    # X=avgSizeAlmond
    # Y=np.log2(avgBSAlmond)
    
    k=10
    
    # k=10
    
    
    # label=['AP-I',"AP-II","AP-III","AP-IV"]
    # label=['MA-I',"MA-II","MA-III","MA-IV",'MA-V']
    # label=['AL-I',"AL-II","AL-III","AL-IV",'AL-V']
    
    svr = SVR(kernel="linear", C=1, gamma="auto")
    fig, axes = plt.subplots()
    preY=svr.fit(X, Y).predict(X)
    axes.plot(
            X,
            preY,
            color='red',
            lw=2,
            label="Linear")  
    
    error=np.abs(Y-preY)
    print('the all linear error is:', np.mean(error))
        
    axes.scatter(X, Y, c='red',marker='+',s=100,label='Best BS')
    
    # preX=[[49],[98],[196],[392],[784]]
    # preX=[[86],[172],[344],[687],[1031],[1375]]
    # preX=[[89],[133],[267],[535],[1071],[2142]]
    preX=X
    
    # for i in range(0, 1):
    #     basicX=X[i*k:(i+1)*k]
    #     basicY=Y[i*k:(i+1)*k]
    #
    #     XUntrain=X[(i+1)*k:len(Y)]
    #     YUntrain=Y[(i+1)*k:len(Y)]
    #
    #     preYUntrain=svr.fit(basicX, basicY).predict(XUntrain)
    #     # print(label[i])
    #     # print('train X:',basicX,basicY)
    #     # print('untrain X',XUntrain,preYUntrain)
    #
    #     error=np.abs(YUntrain-preYUntrain)
    #     print('the test error is:', np.mean(error))
    #
    #     preY=svr.fit(basicX, basicY).predict(preX)
    #     error=np.abs(Y-preY)
    #     # print('real Y and pre Y:',Y, preY)
    #     print('the total error is:', np.mean(error))
    #
    #     axes.plot(
    #         preX,
    #         preY,
    #         color=colorList[i],
    #         lw=2,
    #         linestyle='dashed',
    #         label=label[i])
        
    
    # preList=[2,3]
    
    preList=[1,2,3,4]
    labelList=['AP-I','AP-I/II','AP-I/II/III','AP-I/II/III/IV']
    colorList=['orange','blue','purple','green','black']
    # labelList=['MA-I','MA-I/II','MA-I/II/III','MA-I/II/III/IV']
    # labelList=['AL-I','AL-I/II','AL-I/II/III','AL-I/II/III/IV']
    
    for j in range(0,len(preList)):
        
        index=preList[j]
        
        XUntrain=X[index*k:len(X)]
        YUntrain=Y[index*k:len(Y)]
        
        preYUntrain=svr.fit(X[0:index*k], Y[0:index*k]).predict(XUntrain)
        print(labelList[j])
        # print('train X',X[0:index*k], Y[0:index*k])
        # print('untrain X',XUntrain, preYUntrain)
        
        error=np.abs(YUntrain-preYUntrain)
        print('the test error is:', np.mean(error))
        
        preY=svr.fit(X[0:index*k], Y[0:index*k]).predict(preX)
        error=np.abs(Y-preY)
        # print('real Y :',Y)
        # print('pre Y:', preY)
        print('the total error is:', np.mean(error))
        
        axes.plot(
                preX,
                preY,
                color=colorList[j],
                lw=2,
                linestyle='dashed',
                label=labelList[j])
        
    
    plt.xlabel('Size of training data',fontsize='20')
    plt.xticks(fontsize='20')
    # plt.ylim(0,16)
    plt.ylabel('Log2 of batch size',fontsize='20')
    plt.yticks(fontsize='20')
    plt.legend(ncol=2,loc='upper left',fontsize='14')
    
    plt.savefig('../modelPara/batchsize/log/total170/AP170_Log_Pre.pdf', bbox_inches = "tight")
    
    # plt.savefig('../modelPara/batchsize/log/avg/AL_Log_Pre.pdf', bbox_inches = "tight") 

def mixPrediction():
    svr = SVR(kernel="linear", C=1, gamma="auto")
    
    # fig, axes = plt.subplots()
    model_color = ["red", "green", "black"]
    labelList=['Apple', 'Mango','Almond']
    labelPre=['Mo+Ad->Ae','Ae+Ad->Mo','Ae+Mo->Ad']
    index=[0,1,2]
    # fig, axes = plt.subplots()
    
    # totalX=allFruitsize
    # totalY=allFruitBS
    
    totalX=totalSize
    totalY=totalBS
    
    for i in range(0,3):
        
        print(labelPre[i])
        
        X=avgSize[i]
        # print(X)
        Y=np.log2(avgBS[i])
        
        # plt.scatter(X, Y, c=model_color[i],marker='+', s=100, labe=labelList[i])
        # plt.plot(X, svr.fit(X, Y).predict(X),
        # color=model_color[i],
        # lw=2,
        # linestyle='dashed',
        # label=labelList[i]+'_Lin')
        
        restIndex=[]
        for j in index:
            if i!=j:
                restIndex.append(j)
        print(restIndex)
        
        basicX=[]
        aS=avgSize[restIndex[0]]
        bS=avgSize[restIndex[1]]
        basicX.extend(aS)
        basicX.extend(bS)
        # print(basicX)
        
        basicY=[]
        aB=avgBS[restIndex[0]]
        bB=avgBS[restIndex[1]]
        basicY.extend(aB)
        basicY.extend(bB)
        basicY=np.log2(basicY)
        
        # print(basicY)
        
        preY=svr.fit(basicX, basicY).predict(X)
        
        
        
        # plt.plot(X, preY,
        # color=model_color[i],
        # lw=2,
        # label=labelPre[i])
        # print('Y:',Y)
        # print('preY:',preY)
        error=np.abs(Y-preY)
        print('the test error is:', np.mean(error))
        
        preY=svr.fit(basicX, basicY).predict(totalX)
        error=np.abs(np.log2(totalY)-preY)
        
        # print('totalX:',totalX)
        # print('Y:',np.log2(totalY))
        # print('preY:',preY)
        print('the all error is:', np.mean(error))
        
        
    # plt.xlabel('Size of training data',fontsize='20')
    # plt.xticks(fontsize='20')
    # # plt.ylim(0,16)
    # plt.ylabel('Log2 of batch size',fontsize='20')
    # plt.yticks(fontsize='20')
    # plt.legend(ncol=2,loc='upper left',fontsize='12')
    
    # plt.savefig('../modelPara/batchsize/log/total170/mix170_PreS.pdf', bbox_inches = "tight")
    # plt.savefig('../modelPara/batchsize/log/avg/mix17_PreS.pdf', bbox_inches = "tight")
    
def MA_AL_Pre():
    SG_Oliver=[[91],[1965]]
    
    svr = SVR(kernel="linear", C=1, gamma="auto")
    
    inputSize=[]
    inputBS=[]
    
    # inputSize.extend(avgSizeApple)
    inputSize.extend(avgSizeMango)
    inputSize.extend(avgSizeAlmond)
    
    # inputBS.extend(avgBSApple)
    inputBS.extend(avgBSMango)
    inputBS.extend(avgBSAlmond)
    
    
    
    preY=svr.fit(inputSize, np.log2(inputBS)).predict(SG_Oliver)
    
    print(preY)
    
    
    
    
    
    
    
    
    

def mixedPartitionPre():
    
    svr = SVR(kernel="linear", C=1, gamma="auto")
    
    # index=[[2,2,2],[2,3,3],[3,3,3]]
    
    index=[[3,3,3]]
    
    fig, axes = plt.subplots()
    model_color = ["red", "green", "black"]
    labelList=['(Ae/Mo/Ad)I/II', '(Ae)I/II+(Mo/Ad)I/II/III','(Ae/Mo/Ad)I/II/III']
    fig, axes = plt.subplots()
    
    j=0
    
    k=1
    
    labelFruit=['Apple', 'Mango','Almond']
    
    for i in range(0,3):
        
        X=avgSize[i]
        Y=np.log2(avgBS[i])
        
        # s=100,
        plt.scatter(X, Y, c=model_color[i],marker='+',s=100, label=labelFruit[i])
    
    for i in index:
        
        
        basicX=[]
        basicY=[]
    
        basicX.extend(avgSizeApple[0:i[0]*k])
        basicX.extend(avgSizeMango[0:i[1]*k])
        basicX.extend(avgSizeAlmond[0:i[2]*k])
        
        basicY.extend(np.log2(avgBSApple[0:i[0]*k]))
        basicY.extend(np.log2(avgBSMango[0:i[1]*k]))
        basicY.extend(np.log2(avgBSAlmond[0:i[2]*k]))
        
        print(len(basicY))
        
        testX=[]
        testY=[]
        
        testX.extend(avgSizeApple[i[0]*k:len(avgSizeApple)])
        testX.extend(avgSizeMango[i[1]*k:len(avgSizeMango)])
        testX.extend(avgSizeAlmond[i[2]*k:len(avgSizeAlmond)])
        
        testY.extend(np.log2(avgBSApple[i[0]*k:len(avgSizeApple)]))
        testY.extend(np.log2(avgBSMango[i[1]*k:len(avgSizeMango)]))
        testY.extend(np.log2(avgBSAlmond[i[2]*k:len(avgSizeAlmond)]))
        
        preY=svr.fit(basicX, basicY).predict(testX)
        
        print('testX',testX)
        print('preY:',preY)
        # print('testY:',len(testY),testY)
        
        error=np.abs(testY-preY)
        # print(error)
        print('the test error is:', np.mean(error))
        
        
        preY=svr.fit(basicX, basicY).predict(totalSize)
        
        plt.plot(totalSize, preY,
        color=model_color[j],
        lw=2,
        label=labelList[j])
        
        # print('preY:',preY)
        error=np.abs(np.log2(totalBS)-preY)
        print('the all error is:', np.mean(error))
        
        j=j+1
    
    # plt.xlabel('Size of training data',fontsize='20')
    # plt.xticks(fontsize='20')
    # # plt.ylim(0,16)
    # plt.ylabel('Log2 of batch size',fontsize='20')
    # plt.yticks(fontsize='20')
    # plt.legend(ncol=2,loc='upper left',fontsize='12')
    
    # plt.savefig('../modelPara/batchsize/log/avg/mix17_Pre.pdf', bbox_inches = "tight")
    

if __name__ == '__main__':
    # logApple=[math.log2(x) for x in avgBSApple]
    # print(logApple)
    # print(np.log2(avgBSApple))
    # logRegLinearTotal()
    # singleLogRegLinear()
    # mixPrediction()
    # mixedPartitionPre()
    MA_AL_Pre()
    # file='../datasets/oliver_ISAR/'
    # preTrain_json=file+'olive_train_.json'
    #
    # with open(preTrain_json, 'r') as outfile:
    #     preTrain_list = json.load(outfile)
    # print(len(preTrain_list))

    
    
    
    