from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,Reshape,Permute


# Dense 全连接层
# 运算 output = actication(dot(input,kernel)+bias)

# eg
model = Sequential()
model.add(Dense(32,input_shape=(16,))) #take (*,16)input array,output (*,32) shape array

model.add((Dense(32))) #dont need to specify the size of the input anymore



# keras.layers.core.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# parametres:

# unit 代表改层的输出维度
# actication 激活函数
# use_bias 是否使用偏置项
# kernel_initializer 权值初始化方法
# bias_initializer 偏置向量初始化方法
# kernel_regularizer 施加在权重上的正则项
# bias_regularlizer 施加在偏置上的正则项
# activity_regularlizer 施加在输出上的正则项
# kernel_constrants 施加在约束上的约束项
# bias_constrains 施加在偏置上的约束项



# Actication 层 relu,sigmoid,softmax
ac = Activation('')
# keras.layers.core.Activation(activation)


# Dropout 层为 输入数据施加Dropout。Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合。
d = Dropout(rate=0.1,noise_shape=None,seed=None)
# rate：0~1的浮点数，控制需要断开的神经元的比例
# noise_shape：整数张量，为将要应用在输入上的二值Dropout mask的shape，例如你的输入为(batch_size, timesteps, features)，并且你希望在各个时间步上的Dropout mask都相同，则可传入noise_shape=(batch_size, 1, features)。
# seed：整数，使用的随机数种子


# Flatten 层
# keras.layers.core.Flatten()
# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。

from keras.layers import Convolution2D
# eg
model.add(Convolution2D(64,3,3,border_mode='same',input_shape=(3,32,32)))
# model output (None,64,32,32)
model.add(Flatten())
# model output (None,65536)


