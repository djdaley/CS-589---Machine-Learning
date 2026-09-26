import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# # # Fits one instance of model to data: returns train and test accuracy
def fit(data, seed, model, scale=False):
    # 1. Shuffle
    data = pd.DataFrame(shuffle(data, random_state=seed)).reset_index(drop=True)

    # Features, Classes (X, y) split
    X, y = data.iloc[:,:-1], data.iloc[:,-1]

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=seed)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Scale
    if scale:
        try:
            scaler = Scaler(X_train)
            X_train = pd.DataFrame(scaler.linear_scale(X_train))
            X_test = pd.DataFrame(scaler.linear_scale(X_test))
        except:
            raise TypeError('Data cannot be scaled, likely because it contains non-numerical features')

    model.fit(X_train, y_train)

    # # Training accuracy, Testing accuracy

    return (model.predict(X_train) == y_train).mean(), (model.predict(X_test) == y_test).mean()

# # # Runs fit n times: Returns array of train and test accuracy
def fit_many(data, seeds, model, n, scale=False):
    train_accuracy = []
    test_accuracy = []
    for i in range(n):
        tr, te = fit(data, seeds[i], model, scale)
        train_accuracy.append(tr)
        test_accuracy.append(te)
    return np.array(train_accuracy), np.array(test_accuracy)

class Scaler:
    def __init__(self, X):
        self.min = X.min().to_numpy()
        self.max = X.max().to_numpy()
        self.range = self.max - self.min

    def linear_scale(self, X):
        return (X.to_numpy() - self.min) / self.range