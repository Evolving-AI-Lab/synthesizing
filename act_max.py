#!/usr/bin/env python
import os
os.environ['GLOG_minloglevel'] = '2'  # suprress Caffe verbose prints

import settings
import site
site.addsitedir(settings.caffe_root)

# imports and basic notebook setup
import numpy as np
import math
import os,re,random
import PIL.Image
import sys, subprocess
from IPython.display import clear_output, Image, display
from scipy.misc import imresize
from numpy.linalg import norm
from numpy.testing import assert_array_equal
import scipy.misc, scipy.io
import patchShow

import argparse # parsing arguments

pycaffe_root = settings.caffe_root # substitute your path here
sys.path.insert(0, pycaffe_root)
import caffe

mean = np.float32([104.0, 117.0, 123.0])

fc_layers = ["fc6", "fc7", "fc8", "loss3/classifier", "fc1000", "prob"]
conv_layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]

if settings.gpu:
  caffe.set_mode_gpu() # uncomment this if gpu processing is available

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def get_code(path, layer):
    '''
    Get a code from an image.
    '''

    # set up the inputs for the net: 
    batch_size = 1
    image_size = (3, 227, 227)
    images = np.zeros((batch_size,) + image_size, dtype='float32')

    in_image = scipy.misc.imread(path)
    in_image = scipy.misc.imresize(in_image, (image_size[1], image_size[2]))

    for ni in range(images.shape[0]):
      images[ni] = np.transpose(in_image, (2, 0, 1))

    # RGB to BGR, because this is what the net wants as input
    data = images[:,::-1] 

    # subtract the ImageNet mean
    matfile = scipy.io.loadmat('ilsvrc_2012_mean.mat')
    image_mean = matfile['image_mean']
    topleft = ((image_mean.shape[0] - image_size[1])/2, (image_mean.shape[1] - image_size[2])/2)
    image_mean = image_mean[topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]]
    del matfile
    data -= np.expand_dims(np.transpose(image_mean, (2,0,1)), 0) # mean is already BGR

    #initialize the caffenet to extract the features
    caffenet = caffe.Net(settings.encoder_definition, settings.encoder_path, caffe.TEST)

    # run caffenet and extract the features
    caffenet.forward(data=data)
    feat = np.copy(caffenet.blobs[layer].data)
    del caffenet

    zero_feat = feat[0].copy()[np.newaxis]

    return zero_feat, data

def make_step_decoder(net, x, x0, step_size=1.5, start='pool5', end='fc8'):
    '''Basic gradient ascent step.'''

    src = net.blobs[start] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    # L2 distance between init and target vector
    # Equation (2) in "Visualizing Deep Convolutional Neural Networks Using Natural Pre-Images"
    # http://arxiv.org/pdf/1512.02017.pdf 
    net.blobs[end].diff[...] = (x-x0)
    net.backward(start=end) # back-propagate the inner-product gradient
    g = net.blobs[start].diff.copy()

    # print "g:", g.shape
    grad_norm = norm(g)
    # print " norm decoder: %s" % grad_norm
    # print "max: %s [%.2f]\t obj: %s [%.2f]\t norm: [%.2f]" % (best_unit, fc[best_unit], unit, obj_act, grad_norm)

    # If norm is Nan, skip updating the image
    if math.isnan(grad_norm):
        dst.diff.fill(0.)
        return 1e-12, src.data[:].copy()  
    elif grad_norm == 0:
        dst.diff.fill(0.)
        return 0, src.data[:].copy()

    src.data[:] += step_size/np.abs(g).mean() * g

    # reset objective for next step
    dst.diff.fill(0.)

    return grad_norm, src.data[:].copy()


def make_step_encoder(net, image, xy=0, step_size=1.5, end='fc8', unit=None):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    acts = net.forward(data=image, end=end)

    one_hot = np.zeros_like(dst.data)
    
    # Move in the direction of increasing activation of the given neuron
    if end in fc_layers:
      one_hot.flat[unit] = 1.
    elif end in conv_layers:
      one_hot[:, unit, xy, xy] = 1.
    else:
      raise Exception("Invalid layer type!")
    
    dst.diff[:] = one_hot

    # Get back the gradient at the optimization layer
    diffs = net.backward(start=end, diffs=['data'])
    g = diffs['data'][0]

    # print "g:", g.shape
    grad_norm = norm(g)
    obj_act = 0

    # If grad norm is Nan, skip updating
    if math.isnan(grad_norm):
        dst.diff.fill(0.)
        return 1e-12, src.data[:].copy(), obj_act
    elif grad_norm == 0:
        dst.diff.fill(0.)
        return 0, src.data[:].copy(), obj_act

    # Check if the activation of the given unit is increasing
    if end in fc_layers:
        fc = acts[end][0]
        best_unit = fc.argmax()
        obj_act = fc[unit]
        
    elif end in conv_layers:
        fc = acts[end][0, :, xy, xy]
        best_unit = fc.argmax()
        obj_act = fc[unit]

    print "max: %s [%.2f]\t obj: %s [%.2f]\t norm: [%.2f]" % (best_unit, fc[best_unit], unit, obj_act, grad_norm)

    src.data[:] += step_size/np.abs(g).mean() * g

    # reset objective for next step
    dst.diff.fill(0.)

    return (grad_norm, src.data[:].copy(), obj_act)


def get_shape(data_shape):
    if len(data_shape) == 4:
        # Return (227, 227) from (1, 3, 227, 227) tensor
        size = (data_shape[2], data_shape[3])
    else:
        raise Exception("Data shape invalid.")

    return size


def activation_maximization(encoder, decoder, start_layer, code, octaves, clip=False, debug=False, unit=None, xy=0, upper_bound=None, lower_bound=None, **step_params):

    # Get the input and output sizes
    output_layer = 'deconv0'
    data_shape = encoder.blobs['data'].data.shape
    output_shape = decoder.blobs[output_layer].data.shape

    image_size = get_shape(data_shape)
    output_size = get_shape(output_shape)

    # The top left offset that we start cropping the output image to get the 227x227 image
    topleft = ((output_size[0] - image_size[0])/2, (output_size[1] - image_size[1])/2)

    print "starting optimizing"

    x = None
    src = decoder.blobs[start_layer]
    
    # Make sure the layer size and initial vector size match
    assert_array_equal(src.data.shape, code.shape)

    # src.data is the image x that we optimize
    # code = code[np.newaxis]
    src.data[:] = code.copy()[:]

    # Initialize an empty result
    best_xx = np.zeros(image_size)[np.newaxis]
    best_act = -sys.maxint

    # Save the activation of each image generated
    list_acts = []

    for e,o in enumerate(octaves):
        
        # select layer
        layer = o['layer']

        for i in xrange(o['iter_n']):

            step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']
            
            # 1. pass the pool5 code to decoder to get an image x0
            generated = decoder.forward(feat=src.data[:])
            x0 = generated[output_layer]   # 256x256

            # Crop from 256x256 to 227x227
            cropped_x0 = x0.copy()[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]

            # 2. pass the image x0 to AlexNet to maximize an unit k
            # 3. backprop the activation from AlexNet to the image to get an updated image x
            grad_norm_encoder, x, act = make_step_encoder(encoder, cropped_x0, xy, step_size, end=layer, unit=unit)
            # Convert from BGR to RGB because TV works in RGB
            x = x[:,::-1, :, :]

            # Save this solution if the activation is the highest
            if act > best_act:
              # Don't need to save the highest act
              best_xx = x.copy()
              best_act = act

            # 4. Place the changes in x (227x227) back to x0 (256x256)
            updated_x0 = x0.copy()
            # Crop and convert image from RGB back to BGR
            updated_x0[:,::-1,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]] = x.copy()

            # 5. backprop the image to encoder to get an updated pool5 code
            grad_norm_decoder, updated_code = make_step_decoder(decoder, updated_x0, x0, step_size, start=start_layer, end=output_layer)

            if clip:
              # VAE prior is N(0,1)
              print "clipping ----"
              updated_code = np.clip(updated_code, a_min=-1, a_max=1)

            elif upper_bound != None:
              print "bounding ----"
              updated_code = np.maximum(updated_code, lower_bound) 
              updated_code = np.minimum(updated_code, upper_bound) 

            # Update code
            src.data[:] = updated_code

            # Print x every 10 iterations
            if debug:
                print " ===== %s ===== " % i
                print_x = patchShow.patchShow(x.copy(), in_range=(-120,120))
                name = "./frames/%s.jpg" % str(i).zfill(3)
                scipy.misc.imsave(name, print_x)

                list_acts.append( (name, act) )
    
                print "code --- min: %s -- max: %s" % (np.min(updated_code), np.max(updated_code))
            
            if i % 10 == 0:
                print 'finished step %d in octave %d' % (i,e)
           
            # L2 decay: trying to make the feature vector smaller every iteration
            if o['L2_weight'] > 0 and o['L2_weight'] < 1:
                src.data[:] *= o['L2_weight']

            # Stop if grad is 0
            if grad_norm_decoder == 0:
                print " grad_norm_decoder is 0"
                break
            elif grad_norm_encoder == 0:
                print " grad_norm_encoder is 0"
                break

        print "octave %d image:" % e

    # returning the resulting image
    print " -------------------------"
    print " Result: obj act [%s] " % best_act

    if debug:
      print "Saving list of activations..."
      for p in list_acts:
        name = p[0]
        act = p[1]

        #subprocess.call(["echo \"%s %s\" >> list.txt" % (name, act)], shell=True)
        #print "%s -> %s" % (name, act)
        #subprocess.call(["ls", name])
        write_label(name, act)

    return best_xx

def write_label(filename, act):
    subprocess.call(["convert %s -gravity south -splice 0x10 %s" % (filename, filename)], shell=True)
    subprocess.call(["convert %s -append -gravity Center -pointsize %s label:\"%.2f\" -bordercolor white -border 0x0 -append %s" %
               (filename, 30, act, filename)], shell=True)

def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--unit', metavar='unit', type=int, help='fc8 unit within [0, 999]')
    parser.add_argument('--n_iters', metavar='iter', type=int, default=10, help='Number of iterations')
    parser.add_argument('--L2', metavar='w', type=float, default=1.0, nargs='?', help='L2 weight')
    parser.add_argument('--lr', metavar='lr', type=float, default=2.0, nargs='?', help='Learning rate')
    parser.add_argument('--end_lr', metavar='lr', type=float, default=-1.0, nargs='?', help='Ending Learning rate')
    parser.add_argument('--seed', metavar='n', type=int, default=0, nargs='?', help='Learning rate')
    parser.add_argument('--xy', metavar='n', type=int, default=0, nargs='?', help='Spatial position for conv units')
    parser.add_argument('--opt_layer', metavar='s', type=str, help='Layer at which we optimize a code')
    parser.add_argument('--act_layer', metavar='s', type=str, default="fc8", help='Layer at which we activate a neuron')
    parser.add_argument('--init_file', metavar='s', type=str, default="", help='Init image')
    parser.add_argument('--debug', metavar='b', type=int, default=0, help='Print out the images or not')
    parser.add_argument('--clip', metavar='b', type=int, default=0, help='Clip out the code range to be in N(0,1)')
    parser.add_argument('--bound', metavar='b', type=str, default="", help='The file to an array that is the upper bound for optimization range')
    parser.add_argument('--output_dir', metavar='b', type=str, default=".", help='Output directory for saving results')

    args = parser.parse_args()

    # Default to constant learning rate
    if args.end_lr < 0:
        args.end_lr = args.lr

    # which neuron to visualize
    print "-------------"
    print " unit: %s    xy: %s" % (args.unit, args.xy)
    print " n_iters: %s" % args.n_iters
    print " L2 weight: %s" % args.L2
    print " start learning rate: %s" % args.lr
    print " end learning rate: %s" % args.end_lr
    print " seed: %s" % args.seed
    print " opt_layer: %s" % args.opt_layer
    print " act_layer: %s" % args.act_layer
    print " init_file: %s" % args.init_file
    print " debug: %s" % args.debug
    print " clip: %s" % args.clip
    print " bound: %s" % args.bound
    print " output dir: %s" % args.output_dir
    print "-------------"

    octaves = [
        {
            'layer': args.act_layer,
            'iter_n': args.n_iters,
            'L2_weight': args.L2,
            'start_step_size': args.lr,
            'end_step_size': args.end_lr
        }
    ]

    # networks
    decoder = caffe.Net(settings.decoder_definition, settings.decoder_path, caffe.TEST)
    encoder = caffe.Classifier(settings.encoder_definition, settings.encoder_path,
                           mean = mean, # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

    shape = decoder.blobs['feat'].data.shape

    # Fix the seed
    np.random.seed(args.seed)

    if args.init_file != "":
        start_code, start_image = get_code(args.init_file, args.opt_layer)

        print "Loaded start code: ", start_code.shape
    else:
        start_code = np.random.normal(0, 1, shape)

    n_units = shape[1]

    # Load the activation range
    upper_bound = lower_bound = None

    if args.bound != "":
      #act_range_file = "/home/anh/workspace/upconvnet/act_range_3x/fc6.txt"
      upper_bound = np.loadtxt(args.bound, delimiter=' ', usecols=np.arange(0, n_units), unpack=True)
      upper_bound = upper_bound.reshape(start_code.shape)
      lower_bound = np.zeros(start_code.shape)

    # generate class visualization via octavewise gradient ascent
    output_image = activation_maximization(encoder, decoder, 'feat', start_code, octaves, 
                        clip=args.clip, unit=args.unit, xy=args.xy, debug=args.debug,
                        upper_bound=upper_bound, lower_bound=lower_bound)

    # save image
    filename = "%s/%s_%s_%s_%s_%s__%s.jpg" % (
          args.output_dir,
          args.act_layer, 
          str(args.unit).zfill(4), 
          str(args.n_iters).zfill(2), 
          args.L2, 
          args.lr,
          args.seed
      )

    # Save image
    collage = patchShow.patchShow(output_image, in_range=(-120,120))
    scipy.misc.imsave(filename, collage)

    if args.debug:
      scipy.misc.imsave("./frames/%s.jpg" % str(args.n_iters).zfill(3), collage)

    print "Saved to %s" % filename

if __name__ == '__main__':
    main()
