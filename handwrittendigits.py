import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Dropout

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train/255
x_test=x_test/255

# plt.figure(figsize=(10,5))
# for i in range (20):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i],cmap=plt.cm.binary)
# plt.show()

model=keras.Sequential([Flatten(input_shape=(28,28,1)),
                       Dense(128,activation='relu'),
                       # Dropout(0.3),
                       Dense(10,activation='softmax')])
print(model.summary())

y_train_cat=keras.utils.to_categorical(y_train,10)
y_test_cat=keras.utils.to_categorical(y_test,10)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(x_train,y_train_cat,batch_size=32,epochs=5 ,validation_split=0.2)

model.evaluate(x_test,y_test_cat)

# check prediction
n=1008
x=np.expand_dims(x_test[n],axis=0)
res=model.predict(x)
print (res)
print(f"Digit is: {np.argmax(res)}")
plt.imshow(x_test[n],cmap=plt.cm.binary)
plt.show()

# check whole sample
pred=model.predict(x_test)
pred=np.argmax(pred,axis=1)

print(pred.shape)
print(pred[:30])
print(y_test[:30])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

