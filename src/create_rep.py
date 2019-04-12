import numpy as np
import librosa as lb
import os
from tqdm import tqdm
import glob
from src.config import *
from dotenv import load_dotenv


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)


def extract_features(parent_dir, sub_dirs, file_ext="*.wav", bands=60, frames=41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))):
            sound_clip, s = lb.load(fn)
            label = os.path.basename(fn).split('-')[1]
            for (start, end) in windows(sound_clip, window_size):
                if len(sound_clip[start:end]) == window_size:
                    signal = sound_clip[start:end]
                    melspec = lb.feature.melspectrogram(signal, n_mels=bands)
                    logspec = lb.logamplitude(melspec)
                    logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)
            labels.append(label)
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = lb.feature.delta(features[i, :, :, 0])

    return np.array(features), np.array(labels, dtype=np.int)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def main():
    train_folds = ["fold" + str(i) for i in range(1, 8)]
    test_folds = ["fold8", "fold9", "fold10"]

    data_dir = os.getenv('DATA')
    parent_dir = os.path.join(data_dir, 'audio')

    train_features, train_labels = extract_features(parent_dir, sub_dirs=train_folds)

    np.save(os.path.join(os.getenv('OUTPUT'), train_features_path), train_features, allow_pickle=True)
    np.save(os.path.join(os.getenv('OUTPUT'), train_labels_path), train_labels, allow_pickle=True)

    train_features = np.load(os.path.join(os.getenv('OUTPUT'), train_features_path))
    train_labels = np.load(os.path.join(os.getenv('OUTPUT'), train_labels_path))

    assert train_features.shape[0] == train_labels.shape[0]

    test_features, test_labels = extract_features(parent_dir, sub_dirs=test_folds)

    np.save(os.path.join(os.getenv('OUTPUT'), test_features_path), test_features, allow_pickle=True)
    np.save(os.path.join(os.getenv('OUTPUT'), test_labels_path), test_labels, allow_pickle=True)

    test_features = np.load(os.path.join(os.getenv('OUTPUT'), test_features_path))
    test_labels = np.load(os.path.join(os.getenv('OUTPUT'), test_labels_path))

    assert test_features.shape[0] == test_labels.shape[0]


if __name__ == '__main__':
    load_dotenv()
    main()
