import numpy as np
import tensorflow as tf
import os
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def compare():
    filename=['Resnet','8filters', '16filters', '32filters', '64filters']
    #filename = []
    for model_name in filename:
        loss=[]
        acc=[]
        val_loss=[]
        val_acc=[]
        f_name='/Users/jiaweiqian/Desktop/657A/Assignment4/'+model_name+'.txt'
        with open(f_name) as f:
            for contents in f:
                if ':' in contents:
                    info=contents.split()
                    temp=[]
                    for j in range(len(info)):
                        if ':' in info[j]:
                            temp.append(float(info[j+1]))
                    loss.append(temp[0])
                    acc.append(temp[1])
                    val_loss.append(temp[2])
                    val_acc.append(temp[3])
        epoch=[]
        for i in range(1,len(loss)+1):
            epoch.append(i)
        plt.ylim(0,1.0)
        plt.plot(epoch,loss,'r--',label='train_loss')
        plt.plot(epoch,acc,'b--',label='train_acc')
        plt.plot(epoch,val_loss,'g--',label='val_loss')
        plt.plot(epoch,val_acc,'y--',label='val_acc')
        title=model_name+' model'
        plt.title(title)
        plt.legend()
        plt.show()
        #plot the training accuracy vs loss
        plt.scatter(acc,loss,s=10,color='y',label='training',marker='*')
        plt.scatter(val_acc, val_loss,s=10, color='green', label='validation',marker='s')
        plt.title(title)
        plt.xlabel('accuracy')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

#Load the data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
#Do the normalization for images
train_images, test_images = train_images / 255.0, test_images / 255.0

x_train_used,x_train_not_used,train_labels_used,train_labels_not_used= train_test_split(train_images,train_labels,test_size=0.8)
#Build our models:
CNN_model_1 = tf.keras.Sequential(
    [
    tf.keras.layers.Conv2D(64, (3,3), padding='valid', activation="relu",input_shape=(32, 32, 3)),
    # tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3,3), padding='valid', activation="relu"),
    # tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="sigmoid"),
    tf.keras.layers.Dense(512, activation="sigmoid"),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="relu")
]
)

CNN_model_2 = tf.keras.Sequential(
    [
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu",input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=1),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2), strides=1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="relu")
]
)


CNN_model_3 = tf.keras.Sequential(
    [
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="relu",input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=1),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2), strides=1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu")
]
)
#Test data has already been split as 8:2

CNN_model_1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = CNN_model_1.fit(x_train_used, train_labels_used, batch_size=32, epochs=5,
                    validation_data=(test_images, test_labels))


# pred=CNN_model_1.predict(x_test)
# label=[]
# for j in range(pred.shape[0]):
#     label.append(np.argmax(pred[j,:], axis=0))
# label=np.array(label)
# acc_test=sum(label==y_test)/len(y_test)
# print('The accuracy on test set is '%acc_test)
