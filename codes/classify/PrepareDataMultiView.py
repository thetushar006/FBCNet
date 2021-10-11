import DataLoadingUtils.LoadKUMulti as ld
import numpy as np
from functools import partial
import argparse
import codes.centralRepo.transforms as transforms
import os
import pickle
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="path to numpy data")
    parser.add_argument("--path_to_save", type=str, help="path to folder where the transformed data will be saved")
    parser.add_argument("--move_type", choices=[0, 1], type=int, help="0: MI, 1: realMove")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    path_to_data = args.data_path
    path_to_save = args.path_to_save

    ALL_LABELS = ["all", "Backward", "Cylindrical", "Down", "Forward", "Left", "Lumbrical", "Right", "Spherical",
                     "Up", "twist_Left", "twist_Right"]
    MOVE_TYPE = ["MI", "realMove"]

    subjs = np.array(range(1, 2))
    move_type = MOVE_TYPE[args.move_type]
    print(f"Move type {move_type}")
    labels_idx_for_classif = list(range(1,12))
    labels_for_classif = [ALL_LABELS[idx] for idx in labels_idx_for_classif]
    load_data = partial(ld.LoadKUMulti().get_multi_subject_data, labels_for_classif,
                        move_type, path_to_data)

    filterTransform = {'filterBank': {
        'filtBank': [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]], 'fs': 250,
        'filtType': 'filter'}}
    multiview_transform = transforms.filterBank(**filterTransform['filterBank'])
    for sub_idx in subjs:
        print(f"Subject {sub_idx}")
        x_data, y_labels = load_data([sub_idx])
        print(x_data.shape)
        x_data_mv = torch.zeros((*x_data.shape, len(filterTransform['filterBank']['filtBank'])))
        n_trials, n_channels, n_samples = x_data.shape
        for i in range(n_trials):
            x_trial = x_data[i]
            x_trial_multiview = multiview_transform(x_trial)
            x_data_mv[i] = x_trial_multiview

        eeg_data = {"xdata": x_data_mv,
                    "yLabels": y_labels}

        eeg_fname = f'Sub{sub_idx}_{move_type}.pkl'
        with open(os.path.join(path_to_save, eeg_fname), 'wb') as f:
            pickle.dump(eeg_data, f, protocol=4)