#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import re
import caffe
import numpy as np
import cv2 as cv
import argparse

caffe.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dir', '-d', type=str)
args = parser.parse_args()
print args


def save_tiles(W, fname):
    channels = W.shape[1]
    height = W.shape[2]
    width = W.shape[3]
    print 'channels:', channels
    print 'height:', height
    print 'width:', width

    pad = 1
    side = int(np.ceil(np.sqrt(W.shape[0])))
    output = np.zeros((side * height + (side + 1) * pad,
                       side * width + (side + 1) * pad, channels))
    for sy in range(side):
        for sx in range(side):
            i = sy * side + sx
            if i < W.shape[0]:
                image = W[i].swapaxes(0, 2).swapaxes(0, 1)
                image -= image.min()
                image /= image.max()
                output[sy * height + pad * (sy + 1):
                       (sy + 1) * height + pad * (sy + 1),
                       sx * width + pad * (sx + 1):
                       (sx + 1) * width + pad * (sx + 1), :] = image

    if channels != 3 and channels != 1:
        print output.shape
        for ch in range(channels):
            cv.imwrite('%s_%d.png' % (fname, ch), output[:, :, ch] * 255)
    else:
        cv.imwrite('%s.png' % fname, output * 255)


def search_dirs():
    for fn in glob.glob('*'):
        if os.path.isdir(fn):
            os.chdir(fn)
            fn = '.'
            define = '%s/train_test.prototxt' % fn
            for model in glob.glob('%s/*.caffemodel' % fn):
                num = re.search(ur'_([0-9]+)\.', model).groups()[0]
                if not os.path.exists('%s/weight_%s.png' % (fn, num)):
                    print define, model
                    net = caffe.Net(define, model)
                    conv1_W = net.params['conv1'][0].data
                    save_tiles(conv1_W, '%s/weight_conv1_%s.png' % (fn, num))
            os.chdir('../')

if __name__ == '__main__':
    define = '%s/predict.prototxt' % args.dir
    if not os.path.exists('weights'):
        os.mkdir('weights')
    for model in sorted(glob.glob('%s/snapshots/*.caffemodel' % args.dir)):
        num = re.search(ur'_([0-9]+)\.', model).groups()[0]
        if not os.path.exists('%s/weights/weight_%s.png' % (args.dir, num)):
            net = caffe.Net(define, model, caffe.TEST)
            print define, model

            print net.params
            conv_W = None
            if 'conv1' in net.params:
                conv_W = net.params['conv1'][0].data
            elif 'conv2' in net.params:
                conv_W = net.params['conv2'][0].data
            save_tiles(conv_W, '%s/weights/weight_%s' % (args.dir, num))
