import numpy as np
import pandas as pd
import random
import math
import pickle

NUMERIC_STABILITY = 10**(-7)

def initializeWeights(x):
    w = []
    for num in range(x):
        w.append(random.uniform(-0.0001, 0.0001))
    return np.array(w)


def logisticRegression(x_train, y_train, w, learningRate):
    constant = 0.01
    for epoch in range(int(math.pow(10, 4))):
        y_hat_train = 1 / (1 + (np.exp(-x_train @ w)) + constant)
        gradient = (x_train.T @ (y_hat_train - y_train)) / len(x_train)
        w = w - learningRate * (gradient)
    return w, y_hat_train


def transformPredictions(threshold, predictions):
    YC = []
    for x in predictions:
        if x >= threshold:
            x = 1
        else:
            x = 0
        YC.append(x)
    yc = np.array(YC)
    yc = np.reshape(yc, (len(predictions), 1))
    return yc


def eval_binary_model(Y, yhat=np.array([])):
    # sanity check with sklearn

    # sktruth = Y.flatten().tolist()
    # skpred = yhat

    if type(yhat) == list:
        yhat = np.asarray(yhat)
        yhat = yhat.reshape(-1, 1)
    if Y.shape != yhat.shape:
        Y = Y.reshape(-1, 1)
        yhat = yhat.reshape(-1, 1)
    # logical_tp = np.logical_and(Y, yhat)

    # test_fp = (yhat > Y).sum()
    # test_fn = (yhat < Y).sum()
    tp = np.sum(np.logical_and(Y == 1, yhat == 1))
    fp = np.sum(np.logical_and(Y == 0, yhat == 1))
    fn = np.sum(np.logical_and(Y == 1, yhat == 0))
    tn = np.sum(np.logical_and(Y == 0, yhat == 0))

    precision = tp / (tp + fp + NUMERIC_STABILITY)
    recall = tp / (tp + fn + NUMERIC_STABILITY)
    fscore = (2 * precision * recall) / (precision + recall + NUMERIC_STABILITY)
    # accuracy = np.mean((yhat == Y).sum())
    accuracy = (yhat == Y).sum() / len(Y)
    # metrics = {"precision":np.around(precision,5),
    #             "recall":np.around(recall,5),
    #             "fscore":np.around(fscore,5),
    #             "accuracy":np.around(accuracy,5)}
    metrics = {"precision": precision,
               "recall": recall,
               "fscore": fscore,
               "accuracy": accuracy}
    return metrics

def predict(x_val, w, const):
    pred = 1 / (1 + (np.exp(np.matmul(-x_val, w))) + const)
    return pred



