import numpy as np
import os
from tqdm import tqdm
import random

EXTENSION_SSM = ".ssm"
EXTENSION_LB = ".lb"
EXTENSION_PLY = ".ply"

FORMAT_SSM = "_.ssm.npy"
FORMAT_LB = "_.lb.npy"
FORMAT_PLY = "_.ply.npy"




class UnknownExtensionException(Exception):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return f"Unknown extension"



def get_ssm_lb_ply_file(folder_path, extension):
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.name.endswith(extension):
                yield entry.name


def ToNumpy(extension):
    def ssm2numpy(path, name, start=None):
        start_ = (12 if start == None else start)
        with open(os.path.join(path, name)) as f:
                l = f.readlines()
                return np.array([[np.float32(s) for s in l_.split()] for l_ in l[start_:]], dtype=np.float32).reshape(
                    (int(l[2].split()[0]), 16, 6))
    def lb2numpy(path, name, start=None):
        start_ = (1 if start == None else start)
        with open(os.path.join(path, name)) as f:
            return np.array([float(s) for s in f.readlines()[start_:]], dtype=np.float32)

    def ply2numpy(path, name, start=None):
        start_ = (13 if start == None else start)
        with open(os.path.join(path, name)) as f:
            l = f.readlines()
            return np.array([[np.float32(s) for s in l_.split()] for l_ in l[start_:]], dtype=np.float32)

    if extension == EXTENSION_SSM:
        return ssm2numpy
    elif extension == EXTENSION_LB :
        return lb2numpy
    else :
        return ply2numpy


def save2numpy(array, path, extension):
    np.save(f'{path.rstrip(extension)}_{extension}.npy', array, allow_pickle=False)

def convertAndSave(from_:str, to:str, extension:str, start=None):
    if extension != EXTENSION_LB and extension != EXTENSION_SSM and extension != EXTENSION_PLY:
        raise UnknownExtensionException()
    if not os.path.exists(to):
        os.mkdir(to)
    for file in tqdm(get_ssm_lb_ply_file(from_, extension), total=len(list(get_ssm_lb_ply_file(from_, extension)))):
        save2numpy(ToNumpy(extension)(from_, file, start), os.path.join(to, file), extension)


def get_numpy_file(folder_path, extension=FORMAT_SSM, shuffle=True):
    with os.scandir(folder_path) as entries:
        entries = list(entries)
        if shuffle:
            random.shuffle(entries)
        for entry in entries:
            if entry.name.endswith(extension):
                yield entry.name, f'{entry.name.rstrip(extension)}{FORMAT_LB}'

def nanClean(path, extension):
    for numpy_file in get_numpy_file(path, extension=extension, shuffle=False):
        ssm_file = np.load(os.path.join(path, numpy_file[0]))
        lb_file = np.load(os.path.join(path, numpy_file[1]))
        ssm_mask = np.isnan(ssm_file).any(axis=2).any(axis=1)
        np.save(os.path.join(path, numpy_file[0]), ssm_file[~ssm_mask], allow_pickle=False)
        np.save(os.path.join(path, numpy_file[1]), lb_file[~ssm_mask], allow_pickle=False)









