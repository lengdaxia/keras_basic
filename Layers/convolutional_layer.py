from keras.layers import Conv1D,Conv2D


# Conv1D

# keras.layers.convolutional.Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

# 一维卷积层（即时域卷积），用以在一维输入信号上进行邻域滤波。当使用该层作为首层时，需要提供关键字参数input_shape。
# 例如(10,128)代表一个长为10的序列，序列中每个信号为128向量。而(None, 128)代表变长的128维向量序列。
# 该层生成将输入信号与卷积核按照单一的空域（或时域）方向进行卷积。
# 如果use_bias=True，则还会加上一个偏置项，若activation不为None，则输出为经过激活函数的输出。

# filters：卷积核的数目（即输出的维度）
# kernel_size：整数或由单个整数构成的list/tuple，卷积核的空域或时域窗长度
# strides：整数或由单个整数构成的list/tuple，为卷积的步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
# padding：补0策略，为“valid”, “same” 或“causal”，“causal”将产生因果（膨胀的）卷积，即output[t]不依赖于input[t+1：]。当对不能违反时间顺序的时序信号建模时有用。参考WaveNet: A Generative Model for Raw Audio, section 2.1.。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
# activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
# dilation_rate：整数或由单个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
# use_bias:布尔值，是否使用偏置项
# kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
# bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
# kernel_regularizer：施加在权重上的正则项，为Regularizer对象
# bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
# activity_regularizer：施加在输出上的正则项，为Regularizer对象
# kernel_constraints：施加在权重上的约束项，为Constraints对象
# bias_constraints：施加在偏置上的约束项，为Constraints对象



# Conv2D
# 二维卷积层，即对图像的空域卷积。
# 该层对二维输入进行滑动窗卷积，当使用该层作为第一层时，应提供input_shape参数。
# 例如input_shape = (128,128,3)代表128*128的彩色RGB图像（data_format='channels_last'）

# keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)



# TODO
# SeparableConv2D
# Conv2DTranspose
# Conv3D

# Cropping1D
# Cropping2D
# Cropping3D

# UpSampling1D
# UpSampling2D
# UpSampling3D

# ZeroPadding1D
# ZeroPadding2D
# ZeroPadding3D