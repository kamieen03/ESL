#!/usr/bin/env python3
import numpy as np

class Regressor:
    def __init__(self, train_file, test_file):
        self.Y_train, self.X_train = self._read_file(train_file)
        self.Y_test, self.X_test = self._read_file(test_file)
        self.beta = self.compute_beta()

    def _read_file(self, file_name):
        Y = []
        X = []
        with open(file_name) as f:
            lines = [line.strip() for line in f]
            for l in lines:
                l = l.split(' ')
                y = int(float(l[0]))
                if y not in [2,3]: continue
                x = np.array([float(v) for v in l[1:]])
                Y.append(y)
                X.append(x)
        return np.array(Y), np.array(X)

    def compute_beta(self):
        X = self.X_train
        Y = self.Y_train
        return np.linalg.inv(X.T@X)@X.T@Y

    def classify_train(self):
        Y_hat = self.X_train@self.beta
        Y_hat[Y_hat < 2.5] = 2
        Y_hat[Y_hat >= 2.5] = 3
        return np.sum(Y_hat == self.Y_train)/len(Y_hat)*100

    def classify_test(self):
        Y_hat = self.X_test@self.beta
        Y_hat[Y_hat < 2.5] = 2
        Y_hat[Y_hat >= 2.5] = 3
        return np.sum(Y_hat == self.Y_test)/len(Y_hat)*100


def main():
    train_file = 'data/zip.train'
    test_file = 'data/zip.test'
    r = Regressor(train_file, test_file)
    print(f'{r.classify_train():.4}', end=' ')
    print(f'{r.classify_test():.4}')

if __name__ == '__main__':
    main()
