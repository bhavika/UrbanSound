import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn import metrics
import itertools
from src.create_rep import *
from src.constants import *
from sklearn.metrics import accuracy_score, classification_report

tr_features = np.load(train_features_pickle)
tr_labels = np.load(train_labels_pickle)
tr_labels = one_hot_encode(tr_labels)

ts_features = np.load(test_features_pickle)
ts_labels = np.load(test_labels_pickle)
ts_labels = one_hot_encode(ts_labels)


def SBCNN_Model(field_size, bands, frames, num_channels, num_labels):
    model = Sequential()

    model.add(Convolution2D(24, field_size, field_size, border_mode='same', input_shape=(bands, frames, num_channels)))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(48, field_size, field_size, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(48, field_size, field_size, border_mode='valid'))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(64, W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels, W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Activation('softmax'))

    return model


def run_model(optimizer, n_iter):
    sbcnn = SBCNN_Model(field_size, bands, frames, num_channels, num_labels)

    sbcnn.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    sbcnn.fit(tr_features, tr_labels, batch_size=32, nb_epoch=n_iter)

    y_prob = sbcnn.predict_proba(ts_features, verbose=0)
    y_pred = y_prob.argmax(axis=-1)
    y_true = np.argmax(ts_labels, 1)
    roc = metrics.roc_auc_score(ts_labels, y_prob)

    print("SBCNN with optimizer {} and {} iterations".format(optimizer, n_iter))
    print("ROC:", round(roc, 3))

    score, accuracy = sbcnn.evaluate(ts_features, ts_labels, batch_size=32)
    print("\nOverall accuracy = {:.2f}".format(accuracy))

    print("Classwise accuracy - normalized")
    print(accuracy_score(y_true, y_pred))

    print("Classwise accuracy - unnormalized")
    print(accuracy_score(y_true, y_pred, normalize=True))

    print("Classification Report: {} optimizer, {} iterations".format(optimizer, n_iter))
    print(classification_report(y_true, y_pred, target_names=target_names))


def main():
    sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
    optimizer = [sgd, 'adam']
    n_iter = [100, 200, 300, 500, 2000]

    params = list(itertools.product(optimizer, n_iter))

    for p in params:
        run_model(p[0], p[1])

# if __name__ == '__main__':
#     main()