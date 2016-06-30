#!/usr/bin/env python
'''
Anh Nguyen <anh.ng8@gmail.com>
2016-06-04
'''
import os
os.environ['GLOG_minloglevel'] = '2'  # suprress Caffe verbose prints

import settings
import site
site.addsitedir(settings.caffe_root)
import caffe

import numpy as np
import math, random
import sys, subprocess
from IPython.display import clear_output, Image, display
from scipy.misc import imresize
from numpy.linalg import norm
from numpy.testing import assert_array_equal
import scipy.misc, scipy.io
import patchShow
import argparse # parsing arguments

mean = np.float32([104.0, 117.0, 123.0])

fc_layers = ["fc6", "fc7", "fc8", "loss3/classifier", "fc1000", "prob"]
conv_layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]

if settings.gpu:
  caffe.set_mode_gpu() # uncomment this if gpu processing is available


def get_code(path, layer):
  '''
  Push the given image through an encoder to get a code.
  '''

  # set up the inputs for the net: 
  batch_size = 1
  image_size = (3, 227, 227)
  images = np.zeros((batch_size,) + image_size, dtype='float32')

  in_image = scipy.misc.imread(path)
  in_image = scipy.misc.imresize(in_image, (image_size[1], image_size[2]))

  for ni in range(images.shape[0]):
    images[ni] = np.transpose(in_image, (2, 0, 1))

  # Convert from RGB to BGR
  data = images[:,::-1] 

  # subtract the ImageNet mean
  matfile = scipy.io.loadmat('ilsvrc_2012_mean.mat')
  image_mean = matfile['image_mean']
  topleft = ((image_mean.shape[0] - image_size[1])/2, (image_mean.shape[1] - image_size[2])/2)
  image_mean = image_mean[topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]]
  del matfile
  data -= np.expand_dims(np.transpose(image_mean, (2,0,1)), 0) # mean is already BGR

  # initialize the encoder
  encoder = caffe.Net(settings.encoder_definition, settings.encoder_weights, caffe.TEST)

  # run encoder and extract the features
  encoder.forward(data=data)
  feat = np.copy(encoder.blobs[layer].data)
  del encoder

  zero_feat = feat[0].copy()[np.newaxis]

  return zero_feat, data


def make_step_generator(net, x, x0, start, end, step_size=1):
  '''
  Forward and backward passes through the generator DNN.
  '''

  src = net.blobs[start] # input image is stored in Net's 'data' blob
  dst = net.blobs[end]

  # L2 distance between init and target vector
  net.blobs[end].diff[...] = (x-x0)
  net.backward(start=end)
  g = net.blobs[start].diff.copy()

  grad_norm = norm(g)

  # reset objective after each step
  dst.diff.fill(0.)

  # If norm is Nan, skip updating the image
  if math.isnan(grad_norm):
    return 1e-12, src.data[:].copy()  
  elif grad_norm == 0:
    return 0, src.data[:].copy()

  # Make an update
  src.data[:] += step_size/np.abs(g).mean() * g

  return grad_norm, src.data[:].copy()


def make_step_net(net, end, unit, image, xy=0, step_size=1):
  '''
  Forward and backward passes through the DNN being visualized.
  '''

  src = net.blobs['data'] # input image
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

  grad_norm = norm(g)
  obj_act = 0

  # reset objective after each step
  dst.diff.fill(0.)

  # If grad norm is Nan, skip updating
  if math.isnan(grad_norm):
    return 1e-12, src.data[:].copy(), obj_act
  elif grad_norm == 0:
    return 0, src.data[:].copy(), obj_act

  # Check the activations
  if end in fc_layers:
    fc = acts[end][0]
    best_unit = fc.argmax()
    obj_act = fc[unit]
    
  elif end in conv_layers:
    fc = acts[end][0, :, xy, xy]
    best_unit = fc.argmax()
    obj_act = fc[unit]

  print "max: %4s [%.2f]\t obj: %4s [%.2f]\t norm: [%.2f]" % (best_unit, fc[best_unit], unit, obj_act, grad_norm)

  # Make an update
  src.data[:] += step_size/np.abs(g).mean() * g

  return (grad_norm, src.data[:].copy(), obj_act)


def get_shape(data_shape):

  # Return (227, 227) from (1, 3, 227, 227) tensor
  if len(data_shape) == 4:
    return (data_shape[2], data_shape[3])
  else:
    raise Exception("Data shape invalid.")


def save_image(img, name):
  '''
  Normalize and save the image.
  '''
  img = img[:,::-1, :, :] # Convert from BGR to RGB
  normalized_img = patchShow.patchShow_single(img, in_range=(-120,120))        
  scipy.misc.imsave(name, normalized_img)


def activation_maximization(net, generator, gen_in_layer, gen_out_layer, start_code, params, 
      clip=False, debug=False, unit=None, xy=0, upper_bound=None, lower_bound=None):

  # Get the input and output sizes
  data_shape = net.blobs['data'].data.shape
  generator_output_shape = generator.blobs[gen_out_layer].data.shape

  # Calculate the difference between the input image to the net being visualized
  # and the output image from the generator
  image_size = get_shape(data_shape)
  output_size = get_shape(generator_output_shape)

  # The top left offset that we start cropping the output image to get the 227x227 image
  topleft = ((output_size[0] - image_size[0])/2, (output_size[1] - image_size[1])/2)

  print "Starting optimizing"

  x = None
  src = generator.blobs[gen_in_layer]
  
  # Make sure the layer size and initial vector size match
  assert_array_equal(src.data.shape, start_code.shape)

  # Take the starting code as the input to the generator
  src.data[:] = start_code.copy()[:]

  # Initialize an empty result
  best_xx = np.zeros(image_size)[np.newaxis]
  best_act = -sys.maxint

  # Save the activation of each image generated
  list_acts = []

  for o in params:
    
    # select layer
    layer = o['layer']

    for i in xrange(o['iter_n']):

      step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']
      
      # 1. pass the code to generator to get an image x0
      generated = generator.forward(feat=src.data[:])
      x0 = generated[gen_out_layer]   # 256x256

      # Crop from 256x256 to 227x227
      cropped_x0 = x0.copy()[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]

      # 2. forward pass the image x0 to net to maximize an unit k
      # 3. backprop the gradient from net to the image to get an updated image x
      grad_norm_net, x, act = make_step_net(net=net, end=layer, unit=unit, image=cropped_x0, xy=xy, step_size=step_size)
      
      # Save the solution
      # Note that we're not saving the solutions with the highest activations
      # Because there is no correlation between activation and recognizability
      best_xx = cropped_x0.copy()
      best_act = act

      # 4. Place the changes in x (227x227) back to x0 (256x256)
      updated_x0 = x0.copy()        
      updated_x0[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]] = x.copy()

      # 5. backprop the image to generator to get an updated code
      grad_norm_generator, updated_code = make_step_generator(net=generator, x=updated_x0, x0=x0, 
          start=gen_in_layer, end=gen_out_layer, step_size=step_size)

      # Clipping code
      if clip:
        updated_code = np.clip(updated_code, a_min=-1, a_max=1) # VAE prior is within N(0,1)

      # Clipping each neuron independently
      elif upper_bound is not None:
        updated_code = np.maximum(updated_code, lower_bound) 
        updated_code = np.minimum(updated_code, upper_bound) 

      # L2 on code to make the feature vector smaller every iteration
      if o['L2'] > 0 and o['L2'] < 1:
        updated_code[:] *= o['L2']

      # Update code
      src.data[:] = updated_code

      # Print x every 10 iterations
      if debug:
        print " > %s " % i
        name = "./debug/%s.jpg" % str(i).zfill(3)

        save_image(x.copy(), name)

        # Save acts for later
        list_acts.append( (name, act) )

      # Stop if grad is 0
      if grad_norm_generator == 0:
        print " grad_norm_generator is 0"
        break
      elif grad_norm_net == 0:
        print " grad_norm_net is 0"
        break

  # returning the resulting image
  print " -------------------------"
  print " Result: obj act [%s] " % best_act

  if debug:
    print "Saving list of activations..."
    for p in list_acts:
      name = p[0]
      act = p[1]

      write_label(name, act)

  return best_xx


def write_label(filename, act):
  # Add activation below each image via ImageMagick
  subprocess.call(["convert %s -gravity south -splice 0x10 %s" % (filename, filename)], shell=True)
  subprocess.call(["convert %s -append -gravity Center -pointsize %s label:\"%.2f\" -bordercolor white -border 0x0 -append %s" %
         (filename, 30, act, filename)], shell=True)


def main():

  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--unit', metavar='unit', type=int, help='an unit to visualize e.g. [0, 999]')
  parser.add_argument('--n_iters', metavar='iter', type=int, default=10, help='Number of iterations')
  parser.add_argument('--L2', metavar='w', type=float, default=1.0, nargs='?', help='L2 weight')
  parser.add_argument('--start_lr', metavar='lr', type=float, default=2.0, nargs='?', help='Learning rate')
  parser.add_argument('--end_lr', metavar='lr', type=float, default=-1.0, nargs='?', help='Ending Learning rate')
  parser.add_argument('--seed', metavar='n', type=int, default=0, nargs='?', help='Learning rate')
  parser.add_argument('--xy', metavar='n', type=int, default=0, nargs='?', help='Spatial position for conv units')
  parser.add_argument('--opt_layer', metavar='s', type=str, help='Layer at which we optimize a code')
  parser.add_argument('--act_layer', metavar='s', type=str, default="fc8", help='Layer at which we activate a neuron')
  parser.add_argument('--init_file', metavar='s', type=str, default="None", help='Init image')
  parser.add_argument('--debug', metavar='b', type=int, default=0, help='Print out the images or not')
  parser.add_argument('--clip', metavar='b', type=int, default=0, help='Clip out within a code range')
  parser.add_argument('--bound', metavar='b', type=str, default="", help='The file to an array that is the upper bound for activation range')
  parser.add_argument('--output_dir', metavar='b', type=str, default=".", help='Output directory for saving results')
  parser.add_argument('--net_weights', metavar='b', type=str, default=settings.net_weights, help='Weights of the net being visualized')
  parser.add_argument('--net_definition', metavar='b', type=str, default=settings.net_definition, help='Definition of the net being visualized')

  args = parser.parse_args()

  # Default to constant learning rate
  if args.end_lr < 0:
    args.end_lr = args.start_lr

  # which neuron to visualize
  print "-------------"
  print " unit: %s  xy: %s" % (args.unit, args.xy)
  print " n_iters: %s" % args.n_iters
  print " L2: %s" % args.L2
  print " start learning rate: %s" % args.start_lr
  print " end learning rate: %s" % args.end_lr
  print " seed: %s" % args.seed
  print " opt_layer: %s" % args.opt_layer
  print " act_layer: %s" % args.act_layer
  print " init_file: %s" % args.init_file
  print " clip: %s" % args.clip
  print " bound: %s" % args.bound
  print "-------------"
  print " debug: %s" % args.debug
  print " output dir: %s" % args.output_dir
  print " net weights: %s" % args.net_weights
  print " net definition: %s" % args.net_definition
  print "-------------"

  params = [
    {
      'layer': args.act_layer,
      'iter_n': args.n_iters,
      'L2': args.L2,
      'start_step_size': args.start_lr,
      'end_step_size': args.end_lr
    }
  ]

  # networks
  generator = caffe.Net(settings.generator_definition, settings.generator_weights, caffe.TEST)
  net = caffe.Classifier(args.net_definition, args.net_weights,
               mean = mean, # ImageNet mean
               channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

  # input / output layers in generator
  gen_in_layer = "feat"
  gen_out_layer = "deconv0"

  # shape of the code being optimized
  shape = generator.blobs[gen_in_layer].data.shape

  # Fix the seed
  np.random.seed(args.seed)

  if args.init_file != "None":
    start_code, start_image = get_code(args.init_file, args.opt_layer)

    print "Loaded start code: ", start_code.shape
  else:
    start_code = np.random.normal(0, 1, shape)

  # Load the activation range
  upper_bound = lower_bound = None

  # Set up clipping bounds
  if args.bound != "":
    n_units = shape[1]
    upper_bound = np.loadtxt(args.bound, delimiter=' ', usecols=np.arange(0, n_units), unpack=True)
    upper_bound = upper_bound.reshape(start_code.shape)

    # Lower bound of 0 due to ReLU
    lower_bound = np.zeros(start_code.shape)

  # Optimize a code via gradient ascent
  output_image = activation_maximization(net, generator, gen_in_layer, gen_out_layer, start_code, params, 
            clip=args.clip, unit=args.unit, xy=args.xy, debug=args.debug,
            upper_bound=upper_bound, lower_bound=lower_bound)

  # Save image
  filename = "%s/%s_%s_%s_%s_%s__%s.jpg" % (
      args.output_dir,
      args.act_layer, 
      str(args.unit).zfill(4), 
      str(args.n_iters).zfill(2), 
      args.L2, 
      args.start_lr,
      args.seed
    )

  # Save image
  save_image(output_image, filename)
  print "Saved to %s" % filename

  if args.debug:
    save_image(output_image, "./debug/%s.jpg" % str(args.n_iters).zfill(3))
  

if __name__ == '__main__':
  main()
