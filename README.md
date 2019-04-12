
## Environmental Sound Classification

Classification of urban sounds using deep learning.

### Usage

Requires Python3+ and Spark 2.0+.

`git clone `

` cd <repo> `

`pip install -r requirements.txt`

`chmod +x run.sh`

`./run.sh`



### Troubleshooting

1. On Debian systems, you might run into issues with tensorboard if you don't have `tkinter` installed.
This can be resolved by installing `python3-tk`.

`sudo apt-get install python3-tk`

2. Librosa requires an audio backend for processing WAV files (in UrbanSound8k/audio). If you see errors
that indicate the absence of this backend, you might be missing `libav-tools` on Debian. Install them with -

`sudo apt-get install libav-tools`

