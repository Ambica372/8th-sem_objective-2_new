import zipfile
import numpy as np
import scipy.io as sio

from pipeline.config import EEG_ZIP, EYE_ZIP

def load_mat_from_zip(zip_path):
    data_map = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_name in sorted(z.namelist()):
            if file_name.endswith('.mat'):
                with z.open(file_name) as f:
                    mat = sio.loadmat(f)
                    data_map[file_name] = mat
    return data_map

def extract_trials(mat_dict, prefix):
    trials = []
    for i in range(1, 25):
        key = f"{prefix}{i}"
        if key in mat_dict:
            data = mat_dict[key]
        else:
            found_key = None
            for k in mat_dict.keys():
                if k.startswith(prefix) and str(i) in k:
                    found_key = k
                    break
            if found_key:
                data = mat_dict[found_key]
            else:
                continue

        # Target: normalize all shapes to (time_samples, features)
        # EEG: (62, time, 5) -> (time, 310)
        if data.ndim == 3 and data.shape[0] == 62 and data.shape[2] == 5:
            data = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
        # Eye: (31, time) -> (time, 31)
        elif data.ndim == 2 and data.shape[0] == 31:
            data = data.T
        elif data.ndim == 2 and data.shape[0] == 310:
            data = data.T

        trials.append(data)
    return trials

def generate_labels(trial_lengths):
    base_labels = np.array([
        1, 2, 3, 0, 2, 0, 1, 3, 3, 1, 2, 0, 0, 2, 3, 1, 1, 3, 0, 2, 2, 1, 3, 0
    ])
    labels = []
    for i, length in enumerate(trial_lengths):
        assigned_label = base_labels[i % 24]
        labels.extend([assigned_label] * length)
    return np.array(labels)

def load_data():
    print("Loading EEG data...")
    eeg_mats = load_mat_from_zip(EEG_ZIP)
    
    print("Loading Eye data...")
    eye_mats = load_mat_from_zip(EYE_ZIP)

    print("Aligning and extracting chronological samples...")
    X_eeg_list = []
    X_eye_list = []
    trial_lengths = []
    
    for eeg_file, eye_file in zip(sorted(eeg_mats.keys()), sorted(eye_mats.keys())):
        e_trials = extract_trials(eeg_mats[eeg_file], "de_movingAve")
        i_trials = extract_trials(eye_mats[eye_file], "eye_")
        
        for e_t, i_t in zip(e_trials, i_trials):
            min_len = min(e_t.shape[0], i_t.shape[0])
            X_eeg_list.append(e_t[:min_len, :])
            X_eye_list.append(i_t[:min_len, :])
            trial_lengths.append(min_len)

    X_eeg = np.vstack(X_eeg_list)
    X_eye = np.vstack(X_eye_list)
    y = generate_labels(trial_lengths)

    print("-" * 30)
    print("Data Loaded Successfully:")
    print("EEG shape:", X_eeg.shape)
    print("Eye shape:", X_eye.shape)
    print("Labels shape:", y.shape)
    print("-" * 30)

    return X_eeg, X_eye, y
