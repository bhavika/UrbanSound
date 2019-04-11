from pyspark import SparkContext, SparkConf
from src.constants import *
from src.sbcnn import SBCNN_Model
from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel

conf = SparkConf().setAppName('sbcnn').setMaster('local[0]')
sc = SparkContext(conf=conf)


spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
spark_model.fit(rdd, epochs=20, batch_size=32, verbose=0, validation_split=0.1)

