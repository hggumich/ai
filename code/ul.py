import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
np.set_printoptions(precision=4, suppress=True)

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=100, centers=4,
                  random_state=500, cluster_std=1.25)

model = KMeans(n_clusters=4, random_state=0, n_init=10)

model.fit(x)

y_ = model.predict(x)


print(y_)

plt.figure(figsize=(10,6))
plt.scatter(x[:, 0], x[:, 1], c=y_, cmap='coolwarm')
plt.show();
