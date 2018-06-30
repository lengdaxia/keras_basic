from keras.layers import Dense

layer = Dense(32)

# layer的权重 np.array
ws = layer.get_weights()

# set weights
layer.set_weights(ws)

# 获取当前层配置信息字典，层也可以借由配置信息重构
config = layer.get_config()

# 由config重构new layer
newL = Dense.from_config(config)


# 获取layer的输出张量的shape
layer.output

# 获取layer的输入张量的shape
layer.input

# 输入数据的shape
layer.input_shape

# 输出数据的shape
layer.output_shape


