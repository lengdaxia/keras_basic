from keras.models import Sequential
from keras.layers import Dense,Activation


# keras






# create a model
model = Sequential([
    Dense(32,units=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])


# add() 添加layer
model = Sequential()
model.add(Dense(32,input_shape=(784,)))
model.add(Activation('relu'))


# init the data input shape
# 1
model = Sequential()
model.add(Dense(32,input_dim=784))
# 2
model = Sequential()
model.add(Dense(32,input_shape=(784,)))


# summary
model.summary()



# 编译  三个参数，1 损失函数，2 优化器，3 指标矩阵
# optimizer rmsprop,adagrad,adam，sgd等
# 损失函数 'binary_crossentropy'，'categorical_crossentropy'
# metrics ; ['accuracy','f1-score']



# custom metric
import keras.backend as K
def mean_pred(y_true,y_pred):
    return K.mean(y_true,y_pred)

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy',mean_pred])




# train
import numpy as np
data = np.random.random(1000,100)
labels = np.random.randint(10,size=(1000,1))

model.fit(data,labels,epochs=10,batch_size=32)


