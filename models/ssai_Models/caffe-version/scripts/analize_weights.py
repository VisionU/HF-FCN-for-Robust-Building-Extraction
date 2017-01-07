#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import caffe
import glob
import os
import cv2 as cv

if not os.path.exists('weights'):
    os.mkdir('weights')
for dname in glob.glob('results/*'):
    if not os.path.isdir(dname):
        continue
    for weights in sorted(glob.glob('%s/weights/*' % dname)):
        weight = cv.imread(weights)
    model_name = re.search(ur'results/(.+)_2015', dname).groups()[0]
    cv.imwrite('weights/%s.png' % model_name, weight)
