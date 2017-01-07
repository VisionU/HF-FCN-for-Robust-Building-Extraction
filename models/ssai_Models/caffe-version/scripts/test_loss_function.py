import os
import caffe
import argparse
import cv2 as cv

caffe.set_mode_gpu()
caffe.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', '-d', type=str)
args = parser.parse_args()

os.chdir(args.result_dir)

net = caffe.Net(model_fn, weight_fn, caffe.TEST)
net = caffe.SGDSolver('solver.prototxt')
net.step(1)
data = net.net.blobs['reshape12'].data

img = data[0].transpose([2, 1, 0])

