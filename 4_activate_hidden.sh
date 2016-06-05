#/bin/bash

# Take in an unit number
# if [ "$#" -ne "1" ]; then
#   echo "Provide 1 output channel number at a conv5, e.g. 151"
#   exit 1
# fi

opt_layer=fc6
act_layer=conv5
units="8 55 154 181 45" #"${1}"
xy=6  # spatial location (6, 6)

# Net
net_weights="nets/placesCNN/places205CNN_iter_300000.caffemodel"
net_definition="nets/placesCNN/places205CNN_deploy_updated.prototxt"

# Hyperparam settings for visualizing AlexNet
iters="200"
weights="99"
rates="8.0"
end_lr=1e-10

# Clipping
clip=0
multiplier=3
bound_file=act_range/${multiplier}x/${opt_layer}.txt
init_file="None"

# Debug
debug=0
if [ "${debug}" -eq "1" ]; then
  rm -rf debug
  mkdir debug
fi

# Output dir
output_dir="output"
#rm -rf ${output_dir}
mkdir -p ${output_dir}

list_files=""

# Sweeping across hyperparams
for unit in ${units}; do

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
          
          unit_pad=`printf "%04d" ${unit}`
          f=${output_dir}/${act_layer}_${unit_pad}_${n_iters}_${L2}_${lr}__${seed}.jpg

          # Crop this image according to its receptive field size
          size=163
          offset=32
          convert $f -crop ${size}x${size}+${offset}+${offset} +repage $f           
          convert $f -gravity south -splice 0x10 $f
          convert $f -append -gravity Center -pointsize 30 label:"${unit}" -bordercolor white -border 0x0 -append $f

          # Add to list
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
