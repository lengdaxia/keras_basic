from keras.layers import Dense,Activation,Dropout
from keras.models import Sequential
from keras.optimizers import SGD
import keras

# generate data
import numpy as np
X_train = np.random.random((1000,20))
y_train = keras.utils.to_categorical(np.random.randint(10,size=(1000,1)),num_classes=10)

X_test = np.random.random((100,20))
y_test = keras.utils.to_categorical(np.random.randint(10,size=(100,1)),num_classes=10)


model = Sequential()

# constract layers

model.add(Dense(64,activation='relu',input_dim=20))
model.add(Dropout(0.5))
model.add((Dense(64,activation='relu')))
model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))

# define optimizer
sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
# complie
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

# train
model.fit(x=X_train,y=y_train,epochs=20,batch_size=128)

# score
score = model.evaluate(X_test,y_test,batch_size=128)