import tensorflow as tf
from Script.Convert2numpy import get_numpy_file
import numpy as np
import os

BATCH_DEFAULT = 8192


def tile(df_x, df_y, batch_size):
    """ Complete an incomplete batch and then shuffle ."""
    batches, labels = np.tile(df_x[-1], (batch_size, 1, 1)),  np.tile(df_y[-1], (batch_size, 1, 1))
    index = np.arange(batch_size)
    np.random.shuffle(index)
    return batches[index], labels[index]

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, path, f_input, batch_size=BATCH_DEFAULT, split=1 ,extension="_.ssm.npy", shuffle=True):
        self.batch_size = batch_size
        self.split = split
        self.paths = path
        self.index = 0
        self.gen_input = f_input
        self.shuffle = shuffle # If it's True it will shuffle batches
        self.extension = extension
        self.total_points = self.get_total_point()
        self.generator = get_numpy_file(path, extension=self.extension, shuffle=shuffle)
        self.load_()
        self.diff = 0

    def get_input_shape(self):
        if self.df_x.size != 0:
            return self.df_x.shape

    def load_(self):
        """
        Import a file (either ssm or ply format) along with its corresponding .lb file.
        df_x = for ssm or ply coordinates
        df_y = labels
        It raise an exception when there is no file left
        """
        try:
            x, y = next(self.generator)
            self.df_x = np.load(os.path.join(self.paths, x))
            self.df_y = np.load(os.path.join(self.paths, y))
        except StopIteration:
            self.on_epoch_end()
            raise StopIteration()

    def on_epoch_end(self):
        """On the epoch end it create a new generator"""
        self.generator = get_numpy_file(self.paths, extension=self.extension, shuffle=self.shuffle)
        self.index = 0
        self.diff = 0

    def get_total_point(self):
        """ return the total point on sm or ply files"""
        return np.array([np.load(os.path.join(self.paths, i[0])).shape[0] for i in get_numpy_file(self.paths, extension=self.extension, shuffle=self.shuffle)]).sum()

    def __len__(self):
        """calculate the numbre of batches """
        total_length = self.total_points*self.split
        rest = total_length % self.batch_size
        q = (total_length - rest)/self.batch_size
        return int(q) + int(rest != 0)


    def __getitem__(self, index):
        """return a batches """
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
                batches[-1], labels[-1] = tile(batches[-1], labels[-1], batch_size=self.batch_size)

        batches, labels = np.concatenate(batches, axis=0), np.concatenate(labels, axis=0)
        return self.gen_input(batches), labels.reshape((labels.shape[0], 1))
