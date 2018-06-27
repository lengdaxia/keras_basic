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

# compile parameters
# 1 optimizer 优化器，
# 2 loss 损失函数
# 3 metrics 指标列表
# 4

x = []
y = []
# 4 fit 开始训练model
model.fit(x,y,batch_size=128,epochs=10,verbose=1,callbacks=None,validation_split=0.0,validation_data=None,
          shuffle=True,class_weight=None,sample_weight=None,initial_epoch=0)
# fit parameters
# 1 X-输入数据
# 2 y-标签
# 3 batch_size 每一次给模型训练的样本个数，每次batch回计算一次梯度下降
# 4 epochs 训练终止时候的轮数，每一轮代表整个训练集的样本全部给模型训练
# 5 init_epoch，起始轮数，默认为0，设置之后模型的训练轮数为 epochs-init_epoch
# 6 callBack，list-回调函数
# 7 verbose -输出日志 0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# 8 validation_split 0-1之间，用来指定训练集的一定比例数据作为验证集
# 9 validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。
# 10 class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）
# 11 sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode='temporal'。



# 5 evaluate
model.evaluate(x,y,batch_size=32,verbose=0,sample_weight=None)
# sample_weight：numpy array，含义同fit的同名参数


# 6 predict ,返回np.array
model.predict(x,y,batch_size=32,verbose=0)


# 7 train_on_batch
model.train_on_batch(x, y, class_weight=None, sample_weight=None)
# 8 test_on_batch
model.test_on_batch(x,y,sample_weight=None)
# 9 predict_on_batch
model.predict_on_batch(x)


# 10 fit_generator
model.fit_generator()
# generator：生成器函数，生成器的输出应该为：
#
# 一个形如（inputs，targets）的tuple
# 一个形如（inputs, targets,sample_weight）的tuple。所有的返回值都应该包含相同数目的样本。生成器将无限在数据集上循环。每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束
# steps_per_epoch：整数，当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch
# epochs：整数，数据迭代的轮数
# verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# validation_data：具有以下三种形式之一
#
# 生成验证集的生成器
# 一个形如（inputs,targets）的tuple
# 一个形如（inputs,targets，sample_weights）的tuple
# validation_steps: 当validation_data为生成器时，本参数指定验证集的生成器返回次数
# class_weight：规定类别权重的字典，将类别映射为权重，常用于处理样本不均衡问题。
# sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode='temporal'。
# workers：最大进程数
# max_q_size：生成器队列的最大容量
# pickle_safe: 若为真，则使用基于进程的线程。由于该实现依赖多进程，不能传递non picklable（无法被pickle序列化）的参数到生成器中，因为无法轻易将它们传入子进程中。
# initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。


# 11 evaluate_generator
# 12 predict_generator


