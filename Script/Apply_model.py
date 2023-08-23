import tensorflow as tf
import numpy as np
import os
import random
from Script.Convert2numpy import get_numpy_file
from tqdm import tqdm

GREY_TN = np.array([255, 255, 255, 255])
RED_TP = np.array([255, 0, 0, 255])




class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, path:str, f_input, shuffle=True, extension='_.ssm.npy')->None:
        self.extension =extension
        self.gen_input = f_input
        self.paths = path
        self.shuffle = shuffle
        self.generator = get_numpy_file(path, shuffle=shuffle, extension=self.extension)

    def load_(self):
        x = next(self.generator)[0]
        self.df_x = np.load(os.path.join(self.paths, x))
        return x

    def __len__(self):
        """Count ply files """
        len_ = len(list(self.generator))
        self.generator = get_numpy_file(self.paths, shuffle=self.shuffle, extension=self.extension)
        return len_


    def __getitem__(self, index):
        """return a ply file as numpy array"""
        x = self.load_()
        return self.gen_input(self.df_x), f"{x.rstrip(self.extension)}_.lb.npy"

def get_ply_file(folder_path, extension="_.ssm.npy", shuffle=True):
    with os.scandir(folder_path) as entries:
        entries = list(entries)
        if shuffle:
            random.shuffle(entries)
        for entry in entries:
            if entry.name.endswith(extension):
                yield f"{entry.name.rstrip(extension)}_.lb.npy", f"{entry.name.rstrip(extension)}.ply"


def apply_model(folder_ply, folder_ssm, to_, model, f_input, shuffle=True, extension='_.ssm.npy'):
    """ Apply a model on a generator"""
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus):
        print("Running on GPU")
        tf.config.set_visible_devices(gpus[0], 'GPU')
    for i, name in tqdm(DataGenerator(folder_ssm, f_input, shuffle=False, extension=extension), desc="Prediction"):
        prediction = model.predict(i)
        prediction[np.isnan(prediction)] = 0
        np.save(os.path.join(to_, name), np.where(prediction > 0.5, RED_TP, GREY_TN))
    for x, y in tqdm(get_ply_file(folder_ssm, shuffle=shuffle, extension=extension), desc="create ply files"):
        labels = np.load(os.path.join(to_, x))
        with open(os.path.join(folder_ply, y)) as f:
            metadata = f.readlines()[:13]
            metadata = metadata[:10] + ["property uchar red\n", "property uchar green\n", "property uchar blue\n",
                                        "property uchar alpha\n"] + metadata[10:]
            with open(os.path.join(folder_ply, y)) as f:
                metadata.extend(
                    ["{} {}\n".format(coordi.rstrip('\n'), ' '.join([str(j) for j in labels[i].tolist()])) for i, coordi in enumerate(f.readlines()[13:])])
                with open(os.path.join(to_, f"{y.rstrip('.ply')}_l.ply"), 'w') as f:
                    f.writelines(metadata)