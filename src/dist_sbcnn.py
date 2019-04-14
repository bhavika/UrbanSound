from pyspark import SparkContext, SparkConf
from src.config import field_size, bands, frames, num_channels, num_labels
from src.sbcnn import SBCNN_Model
from keras.optimizers import SGD
from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel
from src.explore import get_data

conf = SparkConf().setAppName('sbcnn').setMaster('local[2]')
sc = SparkContext(conf=conf)


def dist_training(n_iter):
    sbcnn = SBCNN_Model(field_size, bands, frames, num_channels, num_labels)

    sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
    sbcnn.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

    train_arr, train_labels_arr, test_arr, test_labels_arr = get_data()
    rdd = to_simple_rdd(sc, train_arr, train_labels_arr)

    spark_model = SparkModel(sbcnn, frequency='epoch', mode='asynchronous')
    spark_model.fit(rdd, epochs=n_iter, batch_size=32, verbose=0, validation_split=0.1)

    score = spark_model.master_network.evaluate(test_arr, test_labels_arr, verbose=2)
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    dist_training(n_iter=200)
