#!/bin/bash
f=placesCNN.tar.gz

if [ ! -f ${f} ]; then
  wget http://places.csail.mit.edu/model/${f}
fi

tar xvf placesCNN.tar.gz
