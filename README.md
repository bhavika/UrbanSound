
## Environmental Sound Classification

Classification of urban sounds using deep learning.

### Usage

Requires Python3+ and Spark 2.0+.

`git clone git@github.com:bhavika/UrbanSound.git`

`cd UrbanSound`

`virtualenv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

`chmod +x run.sh`

`./run.sh`


### Models

There are 2 models built using Keras and Tensorflow - they are in `src/cnn.py` and `src/sbcnn.py`.
The CNN is a simple 2 layer neural network, whereas `sbcnn.py` contains an implementation of the SBCNN model from
[this paper](https://arxiv.org/pdf/1608.04363.pdf).

You can run train the CNN and predict on 3 folds of the UrbanSound8K dataset using `python3 src/cnn.py`

Similarly, to run the SBCNN - `python3 src/sbcnn.py`.

We've also implemented data-distributed model training setup using [Elephas](https://github.com/maxpumperla/elephas).
This is shown in `src/dist_sbcnn.py`.

We use 2 workers on an m4.2xlarge instance to achieve data-distributed training. If you have Spark set up,
you can test this locally using - `spark-submit src/dist_sbcnn.py`.

The runtime for each of these can be anywhere from a few minutes (15 minutes for `dist_sbcnn.py` with
200 epochs to a few hours (`sbcnn.py` with all configurations).

### Troubleshooting

1. On Debian systems, you might run into issues with tensorboard if you don't have `tkinter` installed.
This can be resolved by installing `python3-tk`.

`sudo apt-get install python3-tk`

2. Librosa requires an audio backend for processing WAV files (in UrbanSound8k/audio). If you see errors
that indicate the absence of this backend, you might be missing `libav-tools` on Debian. Install them with -

`sudo apt-get install libav-tools`

