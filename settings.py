caffe_root = "/home/anh/src/caffe_latest/python" 
gpu = True

# DNN being visualized
net_weights = "nets/caffenet/caffenet.caffemodel"
net_definition = "nets/caffenet/caffenet.prototxt"

# Generator DNN
generator_weights = "nets/upconv/fc6/generator.caffemodel"
generator_definition = "nets/upconv/fc6/generator.prototxt"

# Encoder DNN
encoder_weights = "nets/caffenet/caffenet.caffemodel"
encoder_definition = "nets/caffenet/caffenet.prototxt"
