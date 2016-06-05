caffe_root = "/home/anh/src/caffe_latest/python" 
gpu = True

# DNN being visualized
# These two settings are default, and can be overriden in the act_max.py
net_weights = "nets/caffenet/bvlc_reference_caffenet.caffemodel"
net_definition = "nets/caffenet/caffenet.prototxt"

# Generator DNN
generator_weights = "nets/upconv/fc6/generator.caffemodel"
generator_definition = "nets/upconv/fc6/generator.prototxt"

# Encoder DNN
encoder_weights = "nets/caffenet/bvlc_reference_caffenet.caffemodel"
encoder_definition = "nets/caffenet/caffenet.prototxt"
