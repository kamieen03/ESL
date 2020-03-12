#!/usr/bin/env python3

import sys
import numpy as np
from tqdm import tqdm

class KNN:
    def __init__(self, K, train_file, test_file):
        self.K = K
        self.train = self._read_file(train_file)
        self.test = self._read_file(test_file)

    def _read_file(self, file_name):
        data = []   # list of (y, x) pairs
        with open(file_name) as f:
            lines = [line.strip() for line in f]
            for l in lines:
                l = l.split(' ')
                y = int(float(l[0]))
                if y not in [2,3]: continue
                x = np.array([float(v) for v in l[1:]])
                data.append((y, x))
        return data

    def classify(self, point):
        point = np.array(point)
        assert len(point) == len(self.train[0][1])
        queue = []  # list of (distance, y) pairs

        def insert(queue, dist, y):
            if queue == []:
                queue.append((dist, y))
            for i in range(len(queue)):
                if dist < queue[i][0]:
                    queue.insert(i, (dist, y))
                    break
            if len(queue) > self.K:
                queue.pop()

        def weighted_vote(queue):
            votes = {}
            for dist, y in queue:
                if y not in votes:
                    votes[y] = 0
                votes[y] += 1
            return max(votes.keys(), key = lambda k: votes[k])

        for y, x in self.train:
            dist = np.linalg.norm(point-x)
            if len(queue) < self.K or dist < queue[-1][0]:
                insert(queue, dist, y)

        return weighted_vote(queue)
        
    def classify_train(self):
        right = 0
        done = 0
        for i in range(len(self.train)):
            y, x = self.train[i]
            y_hat = self.classify(x)
            if y == y_hat:
                right += 1
            done += 1
        return right/len(self.train)*100

    def classify_test(self):
        right = 0
        for i in range(len(self.test)):
            y, x = self.test[i]
            y_hat = self.classify(x)
            if y == y_hat:
                right += 1
        return right/len(self.test)*100

def main():
    try:
        K = int(sys.argv[1])
    except:
        K = 1
        print("Running with K=1")
    train_file = 'data/zip.train'
    test_file = 'data/zip.test'
    knn = KNN(K, train_file, test_file)
    print(f'{knn.classify_train():.4}', end=' ')
    print(f'{knn.classify_test():.4}')


if __name__ == '__main__':
    main()
