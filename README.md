## Synthesizing preferred inputs via deep generator networks

This repository contains source code necessary to reproduce some of the main results in the paper:

[Nguyen A](http://anhnguyen.me), [Dosovitskiy A](http://lmb.informatik.uni-freiburg.de/people/dosovits/), [Yosinski J](http://yosinski.com/), [Brox T](http://lmb.informatik.uni-freiburg.de/people/brox/index.en.html), [Clune J](http://jeffclune.com). (2016). ["Synthesizing the preferred inputs for neurons in neural networks via deep generator networks."](http://arxiv.org/abs/1605.09304). arXiv:1605.09304.

**If you use this software in an academic article, please cite:**

    @article{nguyen2016synthesizing,
      title={Synthesizing the preferred inputs for neurons in neural networks via deep generator networks},
      author={Nguyen, Anh and Dosovitskiy, Alexey and Yosinski, Jason and Brox, Thomas and Clune, Jeff},
      journal={arXiv preprint arXiv:1605.09304},
      year={2016}
    }

For more information regarding the paper, please visit www.evolvingai.org/synthesizing

## Setup
This code is built on top of Caffe. You'll need to install the following:
* Install Caffe; follow the official [installation instructions](http://caffe.berkeleyvision.org/installation.html).
* Build the Python bindings for Caffe
* If you have an NVIDIA GPU, you can optionally build Caffe with the GPU option to make it run faster
* Install [ImageMagick](http://www.imagemagick.org/script/binary-releases.php) CLI on your system

## Usage
The main Python file is [act_max.py](act_max.py), which is a standalone Python script; you can pass various command-line arguments to run different experiments. Basically, to synthesize a preferred input for a target neuron *h* (e.g. the “candle” class output neuron), we optimize the hidden code input (red) of a [deep image generator network](https://arxiv.org/abs/1602.02644) to produce an image that highly activates *h*.

<p align="center">
    <img src="http://www.cs.uwyo.edu/~anguyen8/share/160531__arxiv_main_concept.jpg" width=600px>
</p>

We provide here four different examples:

* [1_activate_output.sh](1_activate_output.sh): Optimizing a code to activate an *output* neuron of the [CaffeNet DNN](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet) trained on ImageNet dataset. This script synthesizes images for 5 example neurons and produces this result:

<p align="center">
    <img src="examples/example1.jpg" width=600px>
</p>

* [2_activate_hidden.sh](2_activate_hidden.sh): Optimizing a code to activate a *hidden* neuron of the [CaffeNet DNN](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet) trained on ImageNet dataset. This script synthesizes images for 5 example neurons and produces this result:

* [3_start_from_real_image.sh](3_start_from_real_image.sh):
* [4_activate_output_placesCNN.sh](4_activate_output_placesCNN.sh):

## Licenses
Note that the code in this repository is licensed under MIT License, but, the pre-trained models used by the code have their own licenses. Please carefully check them before use.
* The image generator networks are for non-commercial use only. See their [page](http://lmb.informatik.uni-freiburg.de/resources/software.php) for more.

