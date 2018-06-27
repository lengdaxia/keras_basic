from keras.layers import Dense,Activation,Dropout
from keras.models import Sequential
import numpy as np


# MLP 二分类

# generate data
x_train = np.random.random((1000,20))
y_train = np.random.randint(2,size=(1000,1))
x_test = np.random.random((100,20))
y_test = np.random.randint(2,size=(100,1))

# print(y_train[0:10])


# create model
model = Sequential()

model.add(Dense(64,activation='relu',input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

# compile
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


# train
model.fit(x_train,y_train,epochs=20,batch_size=128)

# score
score = model.evaluate(x_test,y_test,batch_size=128)

print(score)