import tensorflow as tf
import numpy as np
import pandas as pd
from Script.Convert2numpy import get_numpy_file
from Script.DataGenerator import DataGenerator
from tqdm import tqdm
import os

BATCH_SIZE = 8192

def concat_features(path, extension):
    ssm_m, lb_m = [], []
    for numpy_file in get_numpy_file(path, extension=extension, shuffle=False):
        ssm_m.append(np.load(os.path.join(path, numpy_file[0])))
        lb_m.append(np.load(os.path.join(path, numpy_file[1])))
    return np.concatenate(ssm_m, axis=0), np.concatenate(lb_m, axis=0)

class Metrics:
    def __init__(self, predictions, labels):
        self.predictions, self.labels = predictions, labels
        self.TN, self.FN, self.FP, self.TP = self.calculate_metrics()

    def calculate_metrics(self):
        TN_FN_FP_TP = np.zeros((2, 2))
        for pred, label in zip(self.predictions, self.labels):
            TN_FN_FP_TP[pred.squeeze(0), label] += 1
        TN, FN, FP, TP = TN_FN_FP_TP[0, 0], TN_FN_FP_TP[0, 1], TN_FN_FP_TP[1, 0], TN_FN_FP_TP[1, 1],
        return TN, FN, FP, TP

    def precision(self):
        return self.TP / (self.TP + self.FP)

    def recall(self):
        return self.TP / (self.TP + self.FN)

    def mcc(self):
        return (self.TP * self.TN - self.FP * self.FN) / np.sqrt(
            (self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN))

    def f1(self):
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())

    def accuracy(self):
        return (self.TP + self.TN) / (self.TN + self.FN + self.FP + self.TP)

    def iOU(self):
        return self.TP / (self.FN + self.FP + self.TP)


class Evaluation:
    def __init__(self, model, datagen:DataGenerator):
        self.datagen = datagen
        self.model = model
        self.metrics = None
        self.select_cal_unit()

    def select_cal_unit(self):
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus):
            tf.config.set_visible_devices(gpus[0], 'GPU')

    def predict(self):
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus):
            print("Running on GPU")
            tf.config.set_visible_devices(gpus[0], 'GPU')
        self.predictions = []
        self.labels = []
        for i in tqdm(self.datagen, total=len(self.datagen)):
            self.predictions.append(np.where(self.model.predict(i[0]) >= 0.5, 1, 0))
            self.labels.append(i[1])
        self.labels = np.concatenate(self.labels, axis=0).astype(int)
        self.predictions = np.concatenate(self.predictions, axis=0)
        self.metrics = Metrics(np.concatenate(self.predictions, axis=0), np.concatenate(self.labels, axis=0).astype(int))


    def eval_(self):
        if self.metrics != None:
            return pd.DataFrame({"precision": self.metrics.precision(), "recall": self.metrics.recall(), "MCC": self.metrics.mcc(),
                 "F1": self.metrics.f1(), "accuracy": self.metrics.accuracy(), "IoU": self.metrics.iOU()}, index=[0])

