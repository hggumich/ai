import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2 * x ** 2 - x ** 3 / 3

x = np.linspace(-2, 4, 25)
print(x)

y = f(x)
print(y)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'ro')
plt.show()

beta = np.cov(x, y, ddof=0)[0, 1] / np.var(x)
print('beta = ', beta)

alpha = y.mean() - beta * x.mean()
print('alpha = ', alpha)

y_ = alpha + beta * x
print(y_)

MSE = ((y-y_)**2).mean()
print('Mean Squared Error = ', MSE)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'ro', label='sample data')
plt.plot(x, y_, lw=3.0, label='linear regression')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'ro', label='sample data')
for deg in [1, 2, 3]:
    reg = np.polyfit(x, y, deg=deg)
    y_ = np.polyval(reg, x)
    MSE = ((y - y_)**2).mean()
    print(f'deg={deg} | MSE={MSE:.5f}')
    plt.plot(x, np.polyval(reg, x), label=f'deg={deg}')
plt.legend()
plt.show()
print(reg)

# Scikit-learn
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=3 * [256],
                     learning_rate_init=0.03,
                     max_iter=5000)

model.fit(x.reshape(-1, 1), y)

y_ = model.predict(x.reshape(-1, 1))

MSE = ((y - y_)**2).mean()
print('Mean Squared Error = ', MSE)

plt.figure(figsize=(10,6))
plt.plot(x, y, 'ro', label='sample data')
plt.plot(x, y_, lw=3.0, label='dnn estimation')
plt.legend()
plt.show()

# Keras
import tensorflow as tf

tf.random.set_seed(100)

from keras.layers import Dense
from keras.models import Sequential
#Using TensorFlow backend.

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=1))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')

MSE=((y - y_)**2).mean()
print(MSE)

plt.figure(figsize=(10,6))
plt.plot(x, y, 'ro', label='sample data')
for _ in range(1, 6):
    model.fit(x,y, epochs=100, verbose=False)
    y_ = model.predict(x)
    MSE = ((y - y_.flatten())**2).mean()
    print(f'round={_} | MSE={MSE:.5f}')
    plt.plot(x, y_, '--', label=f'round={_}')
plt.legend()
plt.show()

np.random.seed(0)
x = np.linspace(-1, 1)
y = np.random.random(len(x)) * 2 -1

plt.figure(figsize=(10,6))
plt.plot(x, y, 'ro', label='sample data')
for deg in [1, 5, 9, 11, 13, 15]:
    reg = np.polyfit(x, y, deg=deg)
    y_ = np.polyval(reg, x)
    MSE = ((y - y_)**2).mean()
    print(f'deg={deg:2d} | MSE={MSE:.5f}')
    plt.plot(x, np.polyval(reg, x), label=f'deg={deg}')
plt.legend()
plt.show()

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=1))
for _ in range(3):
    model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')
model.summary()

plt.figure(figsize=(10,6))
plt.plot(x, y, 'ro', label='sample data')
for _ in range(1, 8):
    model.fit(x, y, epochs=500, verbose=False)
    y_ = model.predict(x)
    MSE = ((y - y_.flatten())**2).mean()
    print(f'round={_} | MSE={MSE:.5f}')
    plt.plot(x, y_, '--', label=f'round={_}')
plt.legend()
plt.show()  
