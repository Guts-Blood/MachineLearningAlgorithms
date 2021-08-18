from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import math

#Load the data:
data_train=pd.read_csv('/Users/jiaweiqian/Downloads/kmnist_train.csv')
data_test=pd.read_csv('/Users/jiaweiqian/Downloads/kmnist_test.csv')
#Drop the index column
data_train=data_train.drop('Unnamed: 0', axis=1)
data_test=data_test.drop('Unnamed: 0', axis=1)

#Transform then into train and test set
data_train=data_train.to_numpy()
data_test=data_test.to_numpy()
train_X=data_train[:,1:]
train_Y=data_train[:,0]
test_X=data_test[:,1:]
test_Y=data_test[:,0]
#Normalize them
train_X=train_X/255
test_X=test_X/255
#The sigmoid function
def log_sigmoid(z):
    return -np.log(1+np.exp(-z))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(h,Y):
    return (-Y*np.log(h)-(1-Y)*np.log(1-h)).mean()

def predict(X,theta):
    bias=np.ones((X.shape[0],1))
    X=np.concatenate((X,bias),axis=1)
    z=np.dot(X,theta)
    h=log_sigmoid(z)
    return h

def logisticRegression(X,Y,alpha,num_iters):
    model={}
    bias=np.ones((X.shape[0],1))
    X=np.concatenate((X,bias),axis=1)
    theta=np.ones(X.shape[1])
    for step in range(num_iters):
        z=np.dot(X,theta)
        h=sigmoid(z)
        grad=np.dot(X.T,(h-Y))/Y.size
        theta-=alpha*grad
        # if step%1000==0:
        #     z=np.dot(X,theta)
        #     h=sigmoid(z)
        #     print ("{} steps, loss is {}".format(step,loss(h,Y)))
        #     print ("accuracy is {}".format((predict(X[:,:-1],theta,0.5)==Y).mean()))
    model={'theta':theta}
    return theta

#I choose to use 1 vs all training method
classifier=[];
for label in range(10):
    temp_Y=np.zeros((train_Y.shape[0]))
    temp_Y[train_Y==label] = 1
    classifier.append(logisticRegression(train_X,temp_Y,alpha=0.05,num_iters=1000));

Predict=[];
for log_regre in classifier:
    temp_pred=predict(test_X,log_regre);
    Predict.append((temp_pred));

Final_result=[];

for i in range(test_Y.shape[0]):
    max_pro=-100000
    pred_class=0
    for label in range(10):
        if Predict[label][i]>max_pro:
            max_pro=Predict[label][i]
            pred_class=label
    Final_result.append(pred_class)
print("accuracy final_result is %.4f"%(sum(Final_result==test_Y)/test_Y.shape[0]));

'''PCA part'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  #2D绘图库

#Calculate the mean value of data for centralization
def Central_X(X):
    return np.mean(X,axis=0)


#Performe PCA algorithm
def pca(X_data, k):
    #Calculate the mean value for each feature
    center = Central_X(X_data)
    #Get the shape of data
    m, n = np.shape(X_data)
    avgs = np.tile(center, (m, 1))
    data_centralize = X_data - avgs
    covX = np.cov(data_centralize.T)   #Covariance matrix
    featValue, featVec=  np.linalg.eig(covX)  #Calculate the eigenvalue and eigenvector
    Order = np.argsort(-featValue) #Sort the eigenvector with respect to their eigenvalue
    if k > n:
        print ("k must lower than feature number")
        return
    else:
        selectVec = np.matrix(featVec.T[Order[:k]]) #Return the k th biggest eigenvectors
        # finalData = data_centralize * selectVec.T
        # reconData = (finalData * selectVec) + data_centralize
    return selectVec


#输入文件的每行数据都以\t隔开
def loaddata(datafile):
    return np.array(pd.read_csv(datafile,sep=" ",header=-1)).astype(np.float)
def plotBestFit(data1, data2):
    dataArr1 = np.array(data1)
    dataArr2 = np.array(data2)

    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i,0])
        axis_y1.append(dataArr1[i,1])
        axis_x2.append(dataArr2[i,0])
        axis_y2.append(dataArr2[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1'); plt.ylabel('x2');
    plt.savefig("outfile.png")
    plt.show()



#根据数据集data.txt
def main():
    datafile = "data.txt"
    XMat = loaddata(datafile)
    k = 2
    return pca(XMat, k)
if __name__ == "__main__":
    finalData, reconMat = main()
    plotBestFit(finalData, reconMat)




