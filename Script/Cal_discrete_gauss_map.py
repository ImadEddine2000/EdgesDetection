import numpy as np
from Script.Convert2numpy import get_numpy_file
from numba import njit, typed, prange
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

N_NEIGHBORS = 50


@njit(nogil=True)
def get_neighbors(array_obj, point, n_neighbors):
    coordinates = array_obj[:, :3]
    distances = np.sqrt(np.sum((point - coordinates) ** 2, axis=1))
    mask = np.argsort(distances)[:n_neighbors]
    return coordinates[mask]


@njit(nogil=True)
def gauss_map(array_obj, point, n_neighbors=N_NEIGHBORS):
    coordinates = get_neighbors(array_obj, point, n_neighbors=n_neighbors + 1)
    sub_vec = (coordinates - point)[1:]
    dgm = np.zeros((n_neighbors, n_neighbors, 3))
    for i in prange(n_neighbors):
        for j in prange(n_neighbors):
            if i != j:
                nvt = np.cross(sub_vec[i], sub_vec[j])
                norm = np.linalg.norm(nvt)
                dgm[i, j] = point + (nvt / norm if norm != 0 else nvt)
                norm_dmg = np.linalg.norm(dgm[i, j])
                dgm[i, j] = (dgm[i, j] / norm_dmg if norm_dmg != 0 else dgm[i, j])
    return dgm.reshape((1, n_neighbors * n_neighbors, 3))


@njit(nogil=True)
def create_gauss_feature_single(point, array_obj, n_neighbors=N_NEIGHBORS):
    return gauss_map(array_obj, point.reshape((1, 3)), n_neighbors=n_neighbors)


def create_gauss_feature(array_obj, n_neighbors=N_NEIGHBORS):
    with ThreadPoolExecutor() as executor:
        results = typed.List(
            list(executor.map(lambda point: create_gauss_feature_single(point, array_obj, n_neighbors), array_obj)))
    return results


@njit(nogil=True)
def numba_concatante(arr_list):
    num_arrays = len(arr_list)
    total_length = num_arrays * arr_list[0].shape[0]
    result = np.empty((total_length, arr_list[0].shape[1], arr_list[0].shape[2]), dtype=arr_list[0].dtype)
    index = 0
    for arr in arr_list:
        result[index:index + arr.shape[0], :, :] = arr
        index += arr.shape[0]
    return result


def ply2gauss_map(path_from: str, path_to: str, n_neighbors=N_NEIGHBORS, split=1, add_lb=True):
    total = len(list(get_numpy_file(path_from, extension="_.ply.npy", shuffle=False)))
    stop = int(total * split)
    index = 0
    with tqdm(total=stop) as bar:
        for filename in get_numpy_file(path_from, extension="_.ply.npy", shuffle=False):
            array_obj = np.load(os.path.join(path_from, filename[0]))[:, :3]
            if array_obj.shape[0] <= 30000:
                gauss_map_features_list = create_gauss_feature(array_obj, n_neighbors=n_neighbors)
                gauss_map_features = numba_concatante(gauss_map_features_list)
                np.save(os.path.join(path_to, filename[0]), gauss_map_features, allow_pickle=False)
                if add_lb:
                    np.save(os.path.join(path_to, filename[1]), np.load(os.path.join(path_from, filename[1])),
                            allow_pickle=False)
                index += 1
                bar.update(1)
                if index >= stop:
                    break


def oversample_minority_class(gauss_maps, labels, target_ratio=0.5):
    unique_labels, counts = np.unique(labels, return_counts=True)
    majority_class_label = unique_labels[np.argmax(counts)]
    minority_class_label = unique_labels[np.argmin(counts)]

    majority_class_indices = np.where(labels == majority_class_label)[0]
    minority_class_indices = np.where(labels == minority_class_label)[0]

    oversampled_minority_indices = np.random.choice(minority_class_indices,size=int(target_ratio * len(majority_class_indices)), replace=True)

    mask = np.concatenate([majority_class_indices, oversampled_minority_indices])
    np.random.shuffle(mask)
    oversampled_gauss_maps = gauss_maps[mask]
    oversampled_labels = labels[mask]

    return oversampled_gauss_maps, oversampled_labels


def oversample_class(path_from: str, target_ratio=0.5):
    total = len(list(get_numpy_file(path_from, extension="_.ply.npy", shuffle=False)))
    with tqdm(total=total) as bar:
        for filename in get_numpy_file(path_from, extension="_.ply.npy", shuffle=False):
            oversampled_gauss_maps, oversampled_labels = oversample_minority_class(
                np.load(os.path.join(path_from, filename[0])),
                np.load(os.path.join(path_from, filename[1])),
                target_ratio=target_ratio)
            np.save(os.path.join(path_from, filename[0]), oversampled_gauss_maps, allow_pickle=False)
            np.save(os.path.join(path_from, filename[1]), oversampled_labels, allow_pickle=False)
            bar.update(1)