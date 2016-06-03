#/bin/bash

if [ "$#" -ne "1" ]; then
  echo "Missing 1 arg."
  exit 1
fi

#start="${1}"
#end="${start}"

#cd /project/EvolvingAI/anguyen8/x/upconvnet/exp_mean_images

# Go to each class directory
# Find the cluster with the largest number of files
dir="/home/anh/data"

path_labels="/home/anh/src/caffe/data/ilsvrc12/synset_words.txt"
IFS=$'\n' read -d '' -r -a labels < ${path_labels}

opt_layer="fc6"
act_layer="fc8"
#units="9 437 511 736 945"  # fc8
#units="643 10 304 629 945 437 613 846 842 977 626 736 440 531 770 836 899 779 282 162"
units="${1}"

iters=200 #"${2}"
weights="99"
rates="8.0" #"4.0"
debug=0
end_lr=1e-10
#end_lr=${rates} #1e-10
clip=0
multiplier=3
bound_file=act_range/${multiplier}x/${opt_layer}.txt

#rm -rf frames
#mkdir frames

output_dir="test"
rm -rf ${output_dir}
mkdir ${output_dir}

for unit in ${units}; do
  unit_pad=`printf "%04d" ${unit}`
#for unit in {0..999}; do
#for d in `ls -d ${dir}/n*`; 
  category=`echo ${labels[unit]} | cut -d " " -f 1`
  d="${dir}/${category}"

  # Check if directory exists
  #if [ ! -d "${d}" ]; then
  #  echo "Directory not found: ${d}"
  #  break
  #fi

  # Run towards 20 clusters
  #for idx in 13; do
  for idx in {0..0}; do
  #for idx in {0..8}; do
    #mean_file=${d}/mean_${idx}.jpg

    #f=`ls ${d}/${idx}/*.JPEG | head -1`
    #echo ">>> ${f}"

    # fc8 params
    seed=${idx}
    xy=0
    #name="${layer}_${idx}"

    for n_iters in ${iters}; do
      for w in ${weights}; do
        for lr in ${rates}; do

          L2="0.${w}"

          # Optimize images maximizing fc8 unit
          python ./act_max.py \
              --act_layer ${act_layer} \
              --opt_layer ${opt_layer} \
              --unit ${unit} --n_iters ${n_iters} \
              --end_lr ${end_lr} \
              --debug ${debug} \
              --L2 ${L2} --lr ${lr} --seed ${seed} \
              --clip ${clip} \
              --bound ${bound_file} \
              --output_dir ${output_dir}
              #--init_file ${mean_file}
        done
      done
    done
    # Optimize images maximizing fc8 unit
    #./max_layer.sh ${unit} ${name} ${layer} ${xy} ${seed} ${mean_file}
  
  done
done
