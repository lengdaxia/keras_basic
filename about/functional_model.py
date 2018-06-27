from keras.models import Model
from keras.layers import Input,Dense

a = Input(shape=(32,))
b = Dense(32)(a)

#
model = Model(inputs=a,outputs=b)

print(model.outputs)

# common attributes
model.layers
model.inputs
model.outputs

# common methods

# 1
model.compile()

# 2
model.fit()

# 3
model.evaluate()

# 4
model.predict()

# 5
model.train_on_batch()
model.test_on_batch()
model.predict_on_batch()

model.evaluate_generator()
model.predict_generator()
model.fit_generator()