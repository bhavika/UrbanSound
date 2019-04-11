import pandas as pd
import numpy as np
import librosa as lb
import librosa.display as lbdisplay
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.constants import *


def check_data():
    df = pd.read_csv(metadata_csv, sep=',')
    print(df.head())
    print(df['class'].value_counts())
    return df


def generate_samples(dataframe):
    """
        Print out one example from each class
    :return: 
    """
    random_row = np.random.choice(dataframe.index.values, 1)
    sample_row = dataframe.ix[random_row]
    sample_values = sample_row.values
    fn = sample_values[0][0]
    fold = "fold"+str(sample_values[0][5])
    generated_path = audio + fold + "/"+fn
    label = sample_values[0][7]
    return generated_path, label


def load_sound_files(file_paths, sample=True):
    """
    Convert an audio file to numeric array
    :param file_paths: absolute or relative path 
    :param sample: one or full, if one, pass an absolute path for file_paths, else pass link to a folder
    :return: list of lists 
    """
    raw_sounds = []
    labels = []
    df = check_data()
    if sample:
        X, sr = lb.load(file_paths)
        raw_sounds.append(X)
        label = file_paths.split('/')[6].split('-')[1]
        print(label)
        labels.append(label)
    else:
        for fp in tqdm(os.listdir(file_paths)):
            try:
                X, sr = lb.load(file_paths + "/" + fp)
                row = df[df['slice_file_name'] == fp]
                raw_sounds.append(X)
                labels.append(str(row['classID'].values[0]))
            except:
                continue
    return raw_sounds, labels


def display_handler(feature, feature_name, label):
    plt.figure(figsize=(100, 20))
    lbdisplay.specshow(feature, x_axis='time')
    plt.colorbar()
    plt.title(feature_name + " - " +label)
    plt.tight_layout()
    plt.show()


def reshape_input(feature_vector, size):
    """
    Padding smaller feature vectors with zeros for input size compatibility in models. 
    :param feature_vector: 
    :param size: bands * frames
    :return: zero padded feature vector
    """
    try:
        z = np.zeros((1, size - feature_vector[0].shape[1]), dtype=feature_vector[0].dtype)
        feature_vector = np.c_[feature_vector[0], z]
    except IndexError:
        return feature_vector
    return feature_vector


def feature_extraction(sub_dir, sample=True):

    mel_spec_features = []

    if sample is False:
        abs_path = audio + sub_dir
        print("No of files in folder %s are %s" % (sub_dir, len(os.listdir(abs_path))))
        raw_sounds, labels = load_sound_files(abs_path, sample=sample)
    else:
        raw_sounds, labels = load_sound_files(sub_dir, sample=sample)

    for rs in tqdm(raw_sounds):
        print(len(rs))
        melspec = lb.feature.melspectrogram(rs)
        logspec = lb.logamplitude(melspec)
        display_handler(melspec, "Mel Spectrogram", target_names[int(labels[0])])
        logspec = logspec.T.flatten()[:, np.newaxis].T
        mel_spec_features.append(logspec)

    return np.array(mel_spec_features), np.array(labels, dtype=np.int)


def main():
    train_folds = ["fold"+str(i) for i in range(1, 8)]
    test_folds = ["fold"+str(i) for i in range(8, 11)]

    df = check_data()

    for i in range(10):
        path, label = generate_samples(df)
        chan1, labels = feature_extraction(path, sample=True)

if __name__ == '__main__':
    main()