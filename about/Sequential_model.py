# Sequential Model
from keras.models import Sequential
from keras.layers import Dense
# 本质上是一种layer的堆栈 模型，简单易用，但是可定制化程度低.

model = Sequential()

model.add(Dense(32, input_shape=(500,)))
model.add(Dense(10, activation='softmax'))

# common sequential attributes
model_layers = model.layers
print(model_layers)

# common methds
# 1 add
model.add((Dense(32)))

# 2 pop 弹出最后一层的layer
model.pop()

# 3 compile 模型在fit 或者evaluate前必须先编译模型
model.compile(optimizer='rmsprop',
      loss='categorical_crossentropy',
      metrics=['accuracy'])

# 4 fit 开始训练model
