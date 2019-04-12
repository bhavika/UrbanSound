import pandas as pd
import numpy as np
import librosa as lb
import librosa.display as lbdisplay
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv

from src.config import target_names, train_labels_path, train_features_path, test_labels_path, test_features_path
from src.create_rep import one_hot_encode

load_dotenv()

metadata_csv = os.path.join(os.getenv('DATA'), 'metadata/UrbanSound8K.csv')
audio_dir = os.path.join(os.getenv('DATA'), 'audio')


def check_data():
    """
    Print number of samples per class.
    :return: pandas DataFrame
    """
    df = pd.read_csv(metadata_csv, sep=',')
    print(df.head())
    print(df['class'].value_counts())
    return df


def generate_samples(dataframe):
    """
    Return one example selected randomly from each class.
    :return: str, str
    """
    random_row = np.random.choice(dataframe.index.values, 1)
    sample_row = dataframe.ix[random_row]
    sample_values = sample_row.values
    fn = sample_values[0][0]
    fold = "fold"+str(sample_values[0][5])
    generated_path = os.path.join(audio_dir, fold, fn)
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
        label = os.path.basename(file_paths).split('-')[1]
        print(label)
        labels.append(label)
    else:
        for fp in tqdm(os.listdir(file_paths)):
            try:
                X, sr = lb.load(file_paths + "/" + fp)
                row = df[df['slice_file_name'] == fp]
                raw_sounds.append(X)
                labels.append(str(row['classID'].values[0]))
            except IOError as e:
                print(e, fp)
    return raw_sounds, labels


def display_handler(feature, feature_name, label):
    """
    Plot feature against time using librosa's specshow utility.
    :param feature: numpy array
    :param feature_name: str
    :param label: str
    :return:
    """
    plt.figure(figsize=(100, 20))
    lbdisplay.specshow(feature, x_axis='time')
    plt.colorbar()
    plt.title(feature_name + " - " + label)
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
    """
    Extract features for audio files in a directory.
    If sample=True, randomly use one file - else process all files in sub_dir.
    :param sub_dir: str
    :param sample: bool
    :return: numpy array of mel spectrogram features, numpy array of labels
    """
    mel_spec_features = []

    if sample is False:
        abs_path = os.path.join(audio_dir, sub_dir)
        print("No of files in folder {} : {}".format(sub_dir, len(os.listdir(abs_path))))
        raw_sounds, labels = load_sound_files(abs_path, sample=sample)
    else:
        raw_sounds, labels = load_sound_files(sub_dir, sample=sample)

    for rs in tqdm(raw_sounds):
        melspec = lb.feature.melspectrogram(rs)
        logspec = lb.logamplitude(melspec)
        display_handler(melspec, "Mel Spectrogram", target_names[int(labels[0])])
        logspec = logspec.T.flatten()[:, np.newaxis].T
        mel_spec_features.append(logspec)

    return np.array(mel_spec_features), np.array(labels, dtype=np.int)


def get_data():
    """
    Return train-test features and labels from numpy pickles.
    :return: numpy array
    """
    temp_dir = os.getenv('OUTPUT')

    train_arr = np.load(os.path.join(temp_dir, train_features_path))
    train_labels_arr = np.load(os.path.join(temp_dir, train_labels_path))
    train_labels_arr = one_hot_encode(train_labels_arr)

    test_arr = np.load(os.path.join(temp_dir, test_features_path))
    test_labels_arr = np.load(os.path.join(temp_dir, test_labels_path))
    test_labels_arr = one_hot_encode(test_labels_arr)

    return train_arr, train_labels_arr, test_arr, test_labels_arr


def main():
    df = check_data()

    for i in range(10):
        path, label = generate_samples(df)
        mel_spec_features, labels = feature_extraction(path, sample=True)


if __name__ == '__main__':
    main()
