#/bin/bash

# if [ "$#" -ne "1" ]; then
#   echo "Provide 1 output unit number e.g. 8 for art gallery (MIT Places 205)."
#   exit 1
# fi

# Get label for each unit
path_labels="misc/categoryIndex_places205.txt"
IFS=$'\n' read -d '' -r -a labels < ${path_labels}

opt_layer=fc6
act_layer=fc8
units="2 7 8 12 70" #"${1}"
xy=0

# Net
net_weights="nets/placesCNN/places205CNN_iter_300000.caffemodel"
net_definition="nets/placesCNN/places205CNN_deploy.prototxt"

# Hyperparam settings for visualizing AlexNet
iters="200"
weights="99"
rates="8.0"
end_lr=1e-10

# Clipping
clip=0
multiplier=3
bound_file=act_range/${multiplier}x/${opt_layer}.txt
init_file="None" #"images/cat.jpg"

# Debug
debug=0
if [ "${debug}" -eq "1" ]; then
  rm -rf debug
  mkdir debug
fi

# Output dir
output_dir="output"
# rm -rf ${output_dir}
mkdir -p ${output_dir}

list_files=""

# Running optimization across a sweep of hyperparams
for unit in ${units}; do

  label=`echo ${labels[unit]} | cut -d " " -f 1`
  label=${label/_/ }

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
              --init_file ${init_file} \
              --net_weights ${net_weights} \
              --net_definition ${net_definition}

          # Add a category label to each image
          unit_pad=`printf "%04d" ${unit}`
          f=${output_dir}/${act_layer}_${unit_pad}_${n_iters}_${L2}_${lr}__${seed}.jpg
          convert $f -gravity south -splice 0x10 $f
          convert $f -append -gravity Center -pointsize 30 label:"$label" -bordercolor white -border 0x0 -append $f

          list_files="${list_files} ${f}"

        done
      done
    done
  
  done
done

# Make a collage
output_file=${output_dir}/example4.jpg
montage ${list_files} -tile 5x1 -geometry +1+1 ${output_file}
convert ${output_file} -trim ${output_file}
echo "=============================="
echo "Result of example 4: [ ${output_file} ]"

