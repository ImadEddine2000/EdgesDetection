import tensorflow as tf
from Script.Convert2numpy import get_numpy_file
import numpy as np
import json
import os

BATCH_DEFAULT = 8192

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, path, f_input, batch_size=BATCH_DEFAULT, split=1 ,extension="_.ssm.npy", shuffle=True, json_file=None):
        self.batch_size = batch_size
        self.json_file = ((json_file if os.path.exists(json_file) else None) if json_file != None else None)
        self.split = split
        self.paths = path
        self.index = 0
        self.gen_input = f_input
        self.shuffle = shuffle
        self.extension = extension
        self.total_points = self.get_total_point()
        self.generator = get_numpy_file(path, extension=self.extension, shuffle=shuffle)
        self.load_()
        self.diff = 0


    def load_(self):
        x, y = next(self.generator)
        self.df_x = np.load(os.path.join(self.paths, x))
        self.df_y = np.load(os.path.join(self.paths, y))

    def on_epoch_end(self):
        self.generator = get_numpy_file(self.paths, extension=self.extension, shuffle=self.shuffle)
        self.index = 0
        self.diff = 0

    def get_total_point(self):
        if self.json_file == None:
            return np.array([np.load(os.path.join(self.paths, i[0])).shape[0] for i in get_numpy_file(self.paths, extension=self.extension, shuffle=self.shuffle)]).sum()
        else:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
            return data['total_length']

    def __len__(self):
        total_length = self.total_points*self.split
        rest = total_length % self.batch_size
        q = (total_length - rest)/self.batch_size
        return int(q) + 1


    def __getitem__(self, index):
        batches = []
        labels = []
        if (self.index + 1) * self.batch_size + self.diff <= self.df_x.shape[0]:
            batches.append(self.df_x[self.index * self.batch_size+self.diff:(self.index + 1) * self.batch_size+self.diff])
            labels.append(self.df_y[self.index * self.batch_size+self.diff:(self.index + 1) * self.batch_size+self.diff])
            if (self.index + 1) * self.batch_size + self.diff == self.df_x.shape[0] and index < self.__len__() - 1:
                self.load_()
                self.diff = 0
                self.index = 0
            else:
                self.index += 1

        else:

            batches.append(self.df_x[self.index * self.batch_size + self.diff:])
            labels.append(self.df_y[self.index * self.batch_size + self.diff:])
            self.diff += int((self.index + 1) * self.batch_size - self.df_x.shape[0])
            self.index = 0
            try:
                self.load_()
                if self.diff > self.df_x.shape[0]:
                    while True:
                        batches.append(self.df_x)
                        labels.append(self.df_y)
                        self.diff -= self.df_x.shape[0]
                        if self.diff <= self.df_x.shape[0]:
                            break
                labels.append(self.df_y[:self.diff])
                batches.append(self.df_x[:self.diff])
            except StopIteration :
                print(index)
                print(np.concatenate(batches, axis=0).shape)
                pass

        batches, labels = np.concatenate(batches, axis=0), np.concatenate(labels, axis=0)
        return self.gen_input(batches), labels.reshape((labels.shape[0], 1))
