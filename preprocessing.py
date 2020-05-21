import pandas as pd
import numpy as np

data = pd.read_csv('./data/Surgical-deepnet.csv')
data = data.values

np.random.shuffle(data)

X = data.T[:-1]
Y = data.T[-1]
X = X.T
Y = Y.T

X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

X.to_csv('./data/X.csv', index=False)
Y.to_csv('./data/Y.csv', index=False)