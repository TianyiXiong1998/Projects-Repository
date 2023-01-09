import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gc
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow import keras
from math import ceil
import warnings
import linecache
import random
import os
import random
from PIL import Image
warnings.filterwarnings("ignore")
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(192,256,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Flatten())
# model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(8))

model.compile(optimizer='Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



def iter_count(file_name):#count the row
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)

def data_arr(path,batch_size,iter_number):
    tmp = []
    labels = []
    with open(path,'r') as f:
        start = iter_number*batch_size
        end = (iter_number+1)*batch_size
        for i in range(start,end):
            if(linecache.getline(path,i)):
                line = linecache.getline(path,i)
                name = line.split(' ')[0]
                label = int(line.split(' ')[1].strip())
                image = Image.open('/s/parsons/h/proj/vision/data/bmgr/cs510bmgr8/'+ name)
                image.thumbnail((256,256))
                image_arr = np.array(image)

                if(image_arr.shape==(192,256,3)):
                    tmp.append(image_arr)
                    labels.append(label)
    f.close()
    da = (np.array(tmp)/255)
    labels = np.array(labels)
    gc.collect()
    return da,labels

# da,labels = data_arr('/s/chopin/k/grad/xiongty/PA5/test_label.txt',100,58)
# print(da.shape,labels.shape)
def train(train_label_path,test_label_path,batch_size):
    train_count = iter_count(train_label_path)
    test_count = iter_count(test_label_path)
    train_block_size = ceil(train_count/batch_size)
    test_block_size = ceil(test_count/batch_size)

    test_arr,test_labels = data_arr(test_label_path,batch_size,0)
    test_count=0
    #for iter_number_train in range(train_block_size):
    
    acc_list = []
    val_acc_list = []
    for iter_number_train in range(train_block_size):
        print(train_block_size,iter_number_train)
        train_arr,train_labels = data_arr(train_label_path,batch_size,iter_number_train)
        print(train_arr.shape)
        print(train_labels.shape)
        print(test_arr.shape)
        print(test_labels.shape)
            
        history = model.fit(train_arr, train_labels, epochs=1 ,batch_size = 1,validation_data=(test_arr, test_labels))
        print("checkpoint2")
        gc.collect()
        if(iter_number_train%10==0):
            acc_list.append(history.history['accuracy'])
            val_acc_list.append(history.history['val_accuracy'])

    return history,acc_list,val_acc_list


history,acc_list,val_acc_list = train('/s/chopin/k/grad/xiongty/PA5/train_labels.txt','/s/chopin/k/grad/xiongty/PA5/Javelina_test_label.txt',100)

plt.plot(acc_list, label='accuracy')
plt.plot(val_acc_list, label = 'val_accuracy')
plt.xlabel('iter_number')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
plt.savefig("Javelina_Accuracy.jpg")










