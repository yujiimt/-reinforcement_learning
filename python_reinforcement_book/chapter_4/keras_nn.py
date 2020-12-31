import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python import keras as k


dataset = load_boston()


y = dataset.target
x = dataset.data 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

model = k.Sequential([
    k.layers.BatchNormalization(input_shape=(13,)),
    k.layers.Dense(units=13, activation="softplus", kernel_regularizer="l1"),
    k.layers.Dense(units=1)
])

model.compile(loss = "mean_squared_error", optimizer="sgd")
model.fit(X_train, y_train, epochs = 8)

predicts = model.predict(X_test)

result= pd.DataFrame({
    "predict":np.reshape(predicts, (-1,)),
    "actual" : y_test
})

limit = np.max(y_test)
result.plot.scatter(x="actual", y="predict", xlim=(0, limit), ylim=(0, limit))
plt.show()