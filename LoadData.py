import cv2
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from itertools import islice

LIMIT = None

DATA_FOLDER = 'driving_dataset'
TRAIN_FILE = os.path.join(DATA_FOLDER, 'data.txt')

def preprocess(img):
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[:, :, 1], (100, 100))
    return resized

def return_data():
    X = []
    y = []
    features = []

    with open(TRAIN_FILE) as f:
        for line in islice(f, LIMIT):
            path, angle = line.strip().split()
            full_path = os.path.join(DATA_FOLDER, path)
            X.append(full_path)
            y.append(float(angle) * np.pi / 180)

    for i in range(len(X)):
        img = plt.imread(X[i])
        features.append(preprocess(img))
    
    features = np.array(features).astype('float32')
    labels = np.array(y).astype('float32')

    with open('features', 'wb') as f:
        pickle.dump(features, f, protocol=4)
    with open('labels', 'wb') as f:
        pickle.dump(labels, f, protocol=4)

if __name__ == "__main__":
    print('Starting')
    return_data()