### Notes

* The txt files in here are used for clipping the activations of neurons in the input code as described in Sec. 2 (see paper). 
Each neuron is bounded in its own range.

  * `3x` means the upperbound is set as 3 times the standard deviation around the mean activation.
  * If you change to optimize in a different code e.g. `conv5`, rather than the default `fc6`, you'll want to use the corresponding `conv5.txt` file.

* These files are input to the [act_max.py](https://github.com/Evolving-AI-Lab/synthesizing/blob/master/act_max.py) via the `--bound` option.
* If you use your own image generator network (i.e. the prior), you'd need to generate your own txt files containing the activation range computed by running the dataset images through the encoder network.
  
