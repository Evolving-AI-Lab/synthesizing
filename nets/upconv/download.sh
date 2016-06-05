#!/bin/bash
f=release_deepsim_v0.zip

if [ ! -f "${f}" ]; then
  wget http://lmb.informatik.uni-freiburg.de/resources/binaries/arxiv2016_alexnet_inversion_with_gans/release_deepsim_v0.zip
fi

echo "Extracting..."
unzip ${f}

layers="pool5 fc6 fc7"

for layer in ${layers}; do
  mv release_deepsim_v0/${layer}/generator.caffemodel ./${layer}/
done

rm -rf release_deepsim_v0 

echo "Done."
