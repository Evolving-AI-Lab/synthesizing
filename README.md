## Synthesizing preferred inputs via deep generator networks

This repository contains source code necessary to reproduce some of the main results in the paper:

[Nguyen A](http://anhnguyen.me), [Dosovitskiy A](http://lmb.informatik.uni-freiburg.de/people/dosovits/), [Yosinski J](http://yosinski.com/), [Brox T](http://lmb.informatik.uni-freiburg.de/people/brox/index.en.html), [Clune J](http://jeffclune.com). (2016). ["Synthesizing the preferred inputs for neurons in neural networks via deep generator networks."](http://arxiv.org/abs/1605.09304). NIPS 29

**If you use this software in an academic article, please cite:**

    @article{nguyen2016synthesizing,
      title={Synthesizing the preferred inputs for neurons in neural networks via deep generator networks},
      author={Nguyen, Anh and Dosovitskiy, Alexey and Yosinski, Jason band Brox, Thomas and Clune, Jeff},
      journal={NIPS 29},
      year={2016}
    }

For more information regarding the paper, please visit www.evolvingai.org/synthesizing

## Setup

### Installing software
This code is built on top of Caffe. You'll need to install the following:
* Install Caffe; follow the official [installation instructions](http://caffe.berkeleyvision.org/installation.html).
* Build the Python bindings for Caffe
* If you have an NVIDIA GPU, you can optionally build Caffe with the GPU option to make it run faster
* Make sure the path to your `caffe/python` folder in [settings.py](settings.py) is correct
* Install [ImageMagick](http://www.imagemagick.org/script/binary-releases.php) command-line interface on your system.

### Downloading models
You will need to download a few models. There are `download.sh` scripts provided for your convenience.
* The image generation network (Upconvolutional network) from [3]. You can download directly on their [website](https://github.com/anguyen8/upconv_release) or using the provided script `cd nets/upconv && ./download.sh`
* A network being visualized (e.g. from Caffe software package or Caffe Model Zoo). The provided examples use these models:
  * [BVLC reference CaffeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet): `cd nets/caffenet && ./download.sh`
  * [BVLC GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet): `cd nets/googlenet && ./download.sh`
  * [AlexNet CNN trained on MIT Places dataset](http://places.csail.mit.edu/): `cd nets/placesCNN && ./download.sh`

Settings:
* Paths to the downloaded models are in [settings.py](settings.py). They are relative and should work if the `download.sh` scripts run correctly.
* The paths to the model being visualized can be overriden by providing arguments `net_weights` and `net_definition` to [act_max.py](act_max.py).

## Usage
The main algorithm is in [act_max.py](act_max.py), which is a standalone Python script; you can pass various command-line arguments to run different experiments. Basically, to synthesize a preferred input for a target neuron *h* (e.g. the “candle” class output neuron), we optimize the hidden code input (red) of a [deep image generator network](https://arxiv.org/abs/1602.02644) to produce an image that highly activates *h*.

<p align="center">
    <img src="http://www.cs.uwyo.edu/~anguyen8/share/160531__arxiv_main_concept.jpg" width=600px>
</p>

### Examples
We provide here four different examples as a starting point. Feel free to be creative and fork away to produce even cooler results!

[1_activate_output.sh](1_activate_output.sh): Optimizing codes to activate *output* neurons of the [CaffeNet DNN](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet) trained on ImageNet dataset. This script synthesizes images for 5 example neurons. 
* Running `./1_activate_output.sh` produces this result:

<p align="center">
    <img src="examples/example1.jpg" width=600px>
</p>

[2_activate_output_placesCNN.sh](2_activate_output_placesCNN.sh): Optimizing codes to activate *output* neurons of a different network, here [AlexNet DNN](http://places.csail.mit.edu/) trained on [MIT Places205](http://places.csail.mit.edu/) dataset. The same prior used here produces the best images for AlexNet architecture trained on different datasets. It also works on other architectures but the image quality might degrade (see Sec. 3.3 in [our paper](http://arxiv.org/abs/1605.09304)). 
* Running `./2_activate_output_placesCNN.sh` produces this result:

<p align="center">
    <img src="examples/example2.jpg" width=600px>
</p>

[3_start_from_real_image.sh](3_start_from_real_image.sh): Instead of starting from a random code, this example starts from a code of a real image (here, an image of a red bell pepper) and optimizes it to increase the activation of the "bell pepper" neuron. 
* Depending on the hyperparameter settings, one could produce images near or far the initialization code (e.g. ending up with a *green* pepper when starting with a red pepper).
* The `debug` option in the script is enabled allowing one to visualize the activations of intermediate images.
* Running `./3_start_from_real_image.sh` produces this result:

<p align="center">
    <img src="examples/example3.jpg" width=700px>
</p>
<p align="center"><i>Optimization adds more green leaves and a surface below the initial pepper</i></p>


[4_activate_hidden.sh](4_activate_hidden.sh): Optimizing codes to activate *hidden* neurons at layer 5 of the [DeepScene DNN](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet) trained on MIT Places dataset. This script synthesizes images for 5 example neurons. 
* Running `./4_activate_hidden.sh` produces this result:

<p align="center">
    <img src="examples/example4.jpg" width=500px>
</p>
<p align="center"><i>From left to right are units that are semantically labeled by humans in [2] as: <br/>lighthouse, building, bookcase, food, and painting </i></p>

* This result matches the conclusion that object detectors automatically emerge in a DNN trained to classify images of places [2]. See Fig. 6 in [our paper](http://arxiv.org/abs/1605.09304) for more comparison between these images and visualizations produced by [2].

[5_activate_output_GoogLeNet.sh](5_activate_output_GoogLeNet.sh): Here is an example of activating the output neurons of a different architecture, GoogLeNet, trained on ImageNet. Note that the *learning rate* used in this example is different from that in the example 1 and 2 above.
* Running `./5_activate_output_GoogLeNet.sh` produces this result:

<p align="center">
    <img src="examples/example5.jpg" width=600px>
</p>

### Visualizing your own models
* To visualize your own model you should search for the hyperparameter setting that produces the best images for your model.
One simple way to do this is sweeping across different parameters (see code setup in the provided example bash scripts).
* For even better result, one can train an image generator network to invert features from the model being visualized instead of using the provided generator (which is trained to invert CaffeNet). However, training such generator may not be easy for many reasons (e.g. inverting very deep nets like ResNet).

## Licenses
Note that the code in this repository is licensed under MIT License, but, the pre-trained models used by the code have their own licenses. Please carefully check them before use.
* The [image generator networks](https://arxiv.org/abs/1602.02644) (in [nets/upconv/](nets/upconv)) are for non-commercial use only. See their [page](http://lmb.informatik.uni-freiburg.de/resources/software.php) for more.
* See the licenses of the models that you visualize (e.g. [DeepScene CNN](https://people.csail.mit.edu/khosla/papers/iclr2015_zhou.pdf)) before use.

## Questions?
Please feel free to drop [me](http://anhnguyen.me) a line or create github issues if you have questions/suggestions.

## References

[1] Yosinski J, Clune J, Nguyen A, Fuchs T, Lipson H. "Understanding Neural Networks Through Deep Visualization". ICML 2015 Deep Learning workshop.

[2] Zhou B, Khosla A, Lapedriza A, Oliva A, Torralba A. "Object detectors emerge in deep scene cnns". ICLR 2015.

[3] Dosovitskiy A, Brox T. "Generating images with perceptual similarity metrics based on deep networks". arXiv preprint arXiv:1602.02644. 2016
