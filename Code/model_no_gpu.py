#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import csv
import operator
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
from tflearn.layers.normalization import batch_normalization




train_dir = 'C:/Users/Arun/Desktop/ENPM 673/Project 6/train/train'
test_dir = 'C:/Users/Arun/Desktop/ENPM 673/Project 6/test1/test1'
size= 200
alpha = 0.0001
epoch = 1
name = 'data-{}-{}-{}.model'.format(alpha,epoch, '2conv-basic')




def label_image(image): #To split names to dog and  cats 
    name = image.split('.')[0]
    if name =='cat':
        return [1,0]
    elif name=='dog':
        return [0,1]
    
def create_train_data():
    training_data = []
    for im in os.listdir(train_dir):
        label = label_image(im)
        path = os.path.join(train_dir,im)
        new_img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (size,size))
        training_data.append([np.array(new_img),np.array(label)])
    shuffle(training_data) #Shuffle data to avoid overfitting
    np.save('train_set.npy', training_data)
    return training_data


def testing_data():
    testing_data = []
    for im in os.listdir(test_dir):
        path = os.path.join(test_dir,im)
        num = im.split('.')[0]
        new_img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (size,size))
        testing_data.append([np.array(new_img),num])
    np.save('test_set.npy', testing_data)
    return testing_data




def training(mynet,train_set):
    #First layer
    mynet = conv_2d(mynet, 32, 3, activation='relu', padding='same')
    mynet = max_pool_2d(mynet, 3)
    
    #Second layer
    mynet = conv_2d(mynet, 64, 3, activation='relu')
    mynet = max_pool_2d(mynet, 3)
    
    #Third layer
    mynet = conv_2d(mynet, 128, 3, activation='relu')
    mynet = max_pool_2d(mynet, 3)
    mynet = dropout(mynet, 0.8)

    #Fourth layer
    mynet = conv_2d(mynet, 64, 3, activation='relu')
    mynet = max_pool_2d(mynet, 3)
    mynet = batch_normalization(mynet)
       
    #Fifth layer
    #For ouput filter size - 64
    mynet = conv_2d(mynet, 64, 3, activation='relu')
    mynet = max_pool_2d(mynet, 3)
    
    #Sixth layer
    #For ouput filter size - 32
    mynet = conv_2d(mynet, 32, 3, activation='relu')
    mynet = max_pool_2d(mynet, 3)
    mynet = batch_normalization(mynet)

    
    #Fully connected layer with 'relu' activation
    mynet = fully_connected(mynet, 1024, activation='relu')
    
    #drop_out to avoid over-fitting
    mynet = dropout(mynet, 0.8)
    
    #Fully connected layer with 'softmax' activation
    mynet = fully_connected(mynet, 2, activation='softmax')
    mynet = regression(mynet, optimizer='adam', learning_rate = alpha, loss = 'categorical_crossentropy', name='targets')
    
    model = tflearn.DNN(mynet, tensorboard_dir='log')
    
    #Creating 2 new list from train_set and labling them as testing and training sub data sets
    train_sub = train_set[:-2500]    #choosing 22500 sets as train dataset 
    test_sub = train_set[-2500:]     #choosing last 2500 as the test dataset
    
    #for fit
    train_x = np.array([i[0] for i in train_sub]).reshape(-1, size, size, 1)
    train_y = [i[1] for i in train_sub]
    
    #for testing accuracy
    test_x = np.array([i[0] for i in test_sub]).reshape(-1, size, size, 1)
    test_y = [i[1] for i in test_sub]
    
    model.fit({'input': train_x}, {'targets': train_y}, n_epoch=epoch, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=500, show_metric=True)
    model.save(name)    
    return model




train_set = create_train_data()
data =  testing_data()


# # If npy saved before uncomment

# In[21]:


# tf.reset_default_graph()
# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# #loading training data
# train_set = np.load('train_set.npy')
# # test_set = np.load('test_set.npy')  
# data= np.load('test_set.npy') 
# #restoring to curret version
# np.load = np_load_old




mynet = input_data(shape = [None,size,size,1], name='input')
model = training(mynet,train_set)




with open('submission_file.csv','w') as f:
    f.write('id,label\n')

with open('submission_file.csv','a') as f:
    for data in test_set:
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(size,size,1)
        model_out = model.predict([data])[0]
        if(model_out[1]>0.5):
            a=1
        if(model_out[1]<0.5):
            a=0
        f.write('{},{}\n'.format(img_num,a))
    f.close()






