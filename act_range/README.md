### Notes

The txt files in here are used for clipping the activations of neurons in the input code as described in Sec. 2 (see paper). 
Each neuron is bounded in its own range.

* `3x` means the upperbound is set as 3 times the standard deviation around the mean activation.
* If you change to optimize in a different code e.g. `conv5`, rather than the default `fc6`, you'll want to use the corresponding `conv5.txt` file.
