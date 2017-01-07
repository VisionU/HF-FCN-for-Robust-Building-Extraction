#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
import os
import numpy as np
import cv2 as cv
import glob
from collections import defaultdict

result_dir = 'results/Mnih_CNN_Roads-Mini'
test_dir = 'data/mass_roads_mini/test/map'
pad = 24
print result_dir
get_iter = lambda fn: re.search(ur'prediction_([0-9]+)', fn).groups()[0]
get_offset = lambda fn: re.search(ur'pred_([0-9]+)_', fn).groups()[0]
get_img_fn = lambda fn: re.search(ur'pred_[0-9]+_([0-9_]+)\.', fn).groups()[0]

all_predictions = defaultdict(lambda: defaultdict(list))
for dname in glob.glob('%s/*' % result_dir):
    if not os.path.isdir(dname):
        continue

    pred_dirs = sorted(glob.glob('%s/prediction_*' % dname))
    npy_fns = [glob.glob('%s/pred_*.npy' % d) for d in pred_dirs]

    for fn in npy_fns:
        for n in fn:
            n_iter = int(get_iter(n))
            offset = int(get_offset(n))
            img_fn = get_img_fn(n)
            all_predictions[n_iter][img_fn].append({
                'img_fn': img_fn,
                'n_iter': n_iter,
                'offset': offset,
                'npy_fn': n
            })

for n_iter, img_fns in all_predictions.items():
    for predictions in img_fns.values():
        shape = np.load(predictions[0]['npy_fn']).shape
        n_models = len(predictions)
        canvas = np.zeros((shape[0] + 2 * (n_models - 1),
                           shape[1] + 2 * (n_models - 1), shape[2]))
        for prediction in predictions:
            pred = np.load(prediction['npy_fn'])
            offset = prediction['offset']
            h, w, c = pred.shape
            canvas[offset:offset + h, offset:offset + w, :] += pred
        canvas /= n_models

        out_h = n_models + shape[0] - 2 * (n_models - 1)
        out_w = n_models + shape[1] - 2 * (n_models - 1)
        canvas = canvas[n_models - 1:out_h, n_models - 1:out_w, :]

        label_img = cv.imread('%s/%s.tif' % (test_dir, prediction['img_fn']))
        label_img = label_img[pad + n_models - 1:pad + out_h,
                              pad + n_models - 1:pad + out_w, :]

        out_dir = '%s/prediction_%d' % (result_dir, n_iter)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        np.save('%s/pred_%s' %
                (out_dir, prediction['img_fn']), canvas)
        cv.imwrite('%s/pred_%s.png' %
                   (out_dir, prediction['img_fn']), canvas * 255)
        cv.imwrite('%s/label_%s.png' %
                   (out_dir, prediction['img_fn']), label_img * 125)
