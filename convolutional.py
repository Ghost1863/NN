import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train/255
x_test=x_test/255

y_train_cat=keras.utils.to_categorical(y_train,10)
y_test_cat=keras.utils.to_categorical(y_test,10)

x_train=np.expand_dims(x_train,3)
x_test=np.expand_dims(x_test,3)

model=keras.Sequential([
Conv2D(32,(3,3),padding='same',input_shape=(28,28,1),activation='relu'),
MaxPooling2D(pool_size=(2,2),strides=2),
Conv2D(64,(3,3),padding='same',activation='relu'),
MaxPooling2D(pool_size=(2,2),strides=2),
Flatten(),
Dense(128,'relu'),
Dense(10,'softmax')
])

model.compile('adam','categorical_crossentropy',['accuracy'])

print(model.summary())
history=model.fit(x_train,y_train_cat,batch_size=32,epochs=5 ,validation_split=0.2)

model.evaluate(x_test,y_test_cat)