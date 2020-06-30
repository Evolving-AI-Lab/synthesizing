#!/bin/bash
# The following script downloads models made available by the Computer Vision group at University of Freiburg
# http://lmb.informatik.uni-freiburg.de/resources/software.php

# Models are from this paper:
# Dosovitskiy A, Brox T. "Generating images with perceptual similarity metrics based on deep networks". 
# arXiv preprint arXiv:1602.02644. 2016

# Please check their page for licenses and info

# --- Terms of use ---
# All code is provided for research purposes only and without any warranty. 
# Any commercial use requires our consent. When using the code in your research work, 
# you should cite the respective paper. Refer to the readme file in each package to learn how to use the program.

f=release_deepsim_v0.zip

if [ ! -f "${f}" ]; then  
  wget https://lmb.informatik.uni-freiburg.de/resources/binaries/arxiv2016_alexnet_inversion_with_gans/release_deepsim_v0.zip
  # wget http://s.anhnguyen.me/release_deepsim_v0.zip  # Alternative link
fi

echo "Extracting..."
unzip ${f}

layers="pool5 fc6 fc7"

for layer in ${layers}; do
  mv release_deepsim_v0/${layer}/generator.caffemodel ./${layer}/
done

rm -rf release_deepsim_v0 

echo "Done."
