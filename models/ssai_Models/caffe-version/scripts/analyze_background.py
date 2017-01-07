#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

test_dir = 'data/mass_merged/test/map'
for fname in glob.glob('results/*'):
    if 'Buildings' in fname or 'Roads' in fname:
        continue
    if not os.path.isdir(fname):
        continue

    for npy_fn in glob.glob('%s/prediction_500000/*.npy' % fname):
        pred = np.load(npy_fn)
        img_fn = os.path.basename(npy_fn).split('.')[0]
        img_fn = '%s/%s_15.tif' % (test_dir, img_fn.split('_')[1])
        test_map = cv.imread(img_fn, cv.IMREAD_GRAYSCALE)
        background = np.array(test_map == 0)
        background = background[36:-36, 36:-36]
        background_pred = pred[:, :, 0]
        print background_pred
        cv.imshow('back', background_pred)
        cv.waitKey(0)

        true_positives = []
        false_positives = []
        for th in range(0, 255):
            # print 'threshold:', th
            th = 1.0 / 255 * th
            p = np.array(background_pred > th)
            true = float(np.sum(np.array(background == True)))
            false = float(np.sum(np.array(background == False)))
            true_positive = np.sum(
                np.array(p == True) * np.array(background == True))
            true_negative = np.sum(
                np.array(p == False) * np.array(background == False))
            false_positive = np.sum(
                np.array(p == True) * np.array(background == False))
            false_negative = np.sum(
                np.array(p == False) * np.array(background == True))

            true_positives.append(true_positive / true)
            false_positives.append(false_positive / false)

        plt.plot(false_positives, true_positives)
        print npy_fn
    plt.show()

    print fname
