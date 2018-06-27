from keras.models import Sequential
from keras.layers import Dense,Activation

print(' ')
print('*********************  intro  *********************')
print(' ')


# keras



print(' ')
print('*********************  basic keras model creation  *********************')
print(' ')


# create a model
model = Sequential([
    Dense(32,input_dim=100),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])


# add() 添加layer
# model = Sequential()
# model.add(Dense(32,input_shape=(784,)))
# model.add(Activation('relu'))
#
#
# # init the data input shape
# # 1
# model = Sequential()
# model.add(Dense(32,input_dim=784))
# # 2
# model = Sequential()
# model.add(Dense(32,input_shape=(784,)))


# 编译  三个参数，1 损失函数，2 优化器，3 指标矩阵
# optimizer rmsprop,adagrad,adam，sgd等
# 损失函数 'binary_crossentropy'，'categorical_crossentropy'
# metrics ; ['accuracy','f1-score']

# custom metric
import keras.backend as K
def mean_pred(y_true,y_pred):
    return K.mean(y_true,y_pred)

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

# train
import numpy as np
import keras
data = np.random.random((1000,100))
labels = keras.utils.to_categorical(np.random.randint(10,size=(1000,1)),num_classes=10)


model.fit(data,labels,epochs=10,batch_size=32)

# score
score = model.evaluate(data,labels,batch_size=32)



print(' ')
print('********************* two keras models *********************')
print(' ')

# common methods

# 1 summary  打印出模型的概况
model.summary()


# 2 返回模型配置信息
model.get_config()
# eg
config = model.get_config()
model.from_config(config=config)
# or
model = Sequential.from_config(config=config)

# 3 get layer 依据层名或下标获得层对象
layer2 = model.get_layer('dense_2')
print(layer2)

# 4 获取权重
weights = model.get_weights()

# 5 设置model的权重
model.set_weights(weights)

# 6 to_json,返回模型的json字符串（仅包含模型的结构，不包含权重）
from keras.models import model_from_json,model_from_yaml
json_model_string = model.to_json()
model = model_from_json(json_string=json_model_string)

# 7 to_yaml
yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string=yaml_string)


# 8 save weights ,将权重保存到指定路径，文件后缀名 filename.h5
model.save_weights('saved/basic_model_weights.h5')

# 9 load weiths  从HDF5文件中加载权重到当前模型中, 默认情况下模型的结构将保持不变。如果想将权重载入不同的模型（有些层相同）中，则设置by_name=True，只有名字匹配的层才会载入权重
model.load_weights('saved/basic_model_weights.h5')



print(' ')
print('*********************    *********************')
print(' ')


###epoch：中文翻译为时期。
# 一个时期 = 所有训练样本的一个正向传递和一个反向传递。
#
# 深度学习中经常看到epoch、 iteration和batchsize，下面按自己的理解说说这三个的区别：
#
# （1）batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练；
# （2）iteration：1个iteration等于使用batchsize个样本训练一次；
# （3）epoch：1个epoch等于使用训练集中的全部样本训练一次；
#
# 举个例子，训练集有1000个样本，batchsize=10，那么：
# 训练完整个样本集需要：
# 100次iteration，1次epoch。
###