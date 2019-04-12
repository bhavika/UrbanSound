#!/bin/bash

wget https://serv.cusp.nyu.edu/files/jsalamon/datasets/UrbanSound8K.tar.gz
tar -xvf UrbanSound8K.tar.gz UrbanSound8K

mkdir output

python3 src/create_rep.py
python3 src/cnn.py
python3 src/sbcnn.py
