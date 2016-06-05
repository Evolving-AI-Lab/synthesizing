#!/bin/bash

f=bvlc_googlenet.caffemodel

if [ ! -f "${f}" ]; then 
  echo "Downloading..."
  wget http://dl.caffe.berkeleyvision.org/${f}
fi

echo "Done."
