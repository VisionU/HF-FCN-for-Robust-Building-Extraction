#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import caffe

caffe.set_mode_gpu()
caffe.set_device(8)

model_fn = 'solver.prototxt'
solver = caffe.SGDSolver(model_fn)

for t in range(2):
    solver.step(1)

output = solver.net.forward()
print output
#
# for i in range(128):
#     data = output['data'][i]
#     label = output['label'][i]
#
#     data = data.transpose([1, 2, 0])
#     label = label.transpose([1, 2, 0])
#
#     ldata = np.copy(data)
#     ldata[data.shape[0] / 2 - label.shape[0] / 2:
#           data.shape[0] / 2 + label.shape[0] / 2,
#           data.shape[1] / 2 - label.shape[0] / 2:
#           data.shape[1] / 2 + label.shape[1] / 2, :] = label * 255
#     data = np.hstack([data, ldata])
#     cv.imwrite('%d.png' % i, data)
