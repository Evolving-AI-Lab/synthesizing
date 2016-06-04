#/bin/bash

if [ "$#" -ne "1" ]; then
  echo "Missing 1 arg."
  exit 1
fi

# Go to each class directory
# Find the cluster with the largest number of files
dir="/home/anh/data"

path_labels="/home/anh/src/caffe/data/ilsvrc12/synset_words.txt"
IFS=$'\n' read -d '' -r -a labels < ${path_labels}

opt_layer=fc6
act_layer=fc8
#units="643 10 304 629 945 437 613 846 842 977 626 736 440 531 770 836 899 779 282 162"
units="${1}"
xy=0

# Hyperparam settings for AlexNet DNNs
iters="200"
weights="99"
rates="8.0"
end_lr=1e-10

# Clipping
clip=0
multiplier=3
bound_file=act_range/${multiplier}x/${opt_layer}.txt
init_file="images/cat.jpg"

# Debug
debug=0
if [ "${debug}" -eq "1" ]; then
  rm -rf debug
  mkdir debug
fi

# Output dir
output_dir="test"
rm -rf ${output_dir}
mkdir ${output_dir}

# Running optimization across a sweep of hyperparams
for unit in ${units}; do
  unit_pad=`printf "%04d" ${unit}`
  # category=`echo ${labels[unit]} | cut -d " " -f 1`

  for seed in {0..0}; do
  #for seed in {0..8}; do

    for n_iters in ${iters}; do
      for w in ${weights}; do
        for lr in ${rates}; do

          L2="0.${w}"

          # Optimize images maximizing fc8 unit
          python ./act_max.py \
              --act_layer ${act_layer} \
              --opt_layer ${opt_layer} \
              --unit ${unit} \
              --xy ${xy} \
              --n_iters ${n_iters} \
              --start_lr ${lr} \
              --end_lr ${end_lr} \
              --L2 ${L2} \
              --seed ${seed} \
              --clip ${clip} \
              --bound ${bound_file} \
              --debug ${debug} \
              --output_dir ${output_dir} \
              --init_file ${init_file}
        done
      done
    done
  
  done
done
