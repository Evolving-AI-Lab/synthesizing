#!/bin/bash
f=release_deepsim_v0.zip

if [ ! -f "${f}" ]; then
  wget http://lmb.informatik.uni-freiburg.de/resources/binaries/arxiv2016_alexnet_inversion_with_gans/release_deepsim_v0.zip
fi

unzip ${f}
mv release_deepsim_v0/fc6/generator.caffemodel ./

rm -rf release_deepsim_v0 
