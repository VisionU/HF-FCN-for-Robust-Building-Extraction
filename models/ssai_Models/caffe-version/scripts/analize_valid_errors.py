#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import sys
if sys.platform.startswith('linux'):
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

result_dir = 'results/Mnih_CNN_Asym_2015-03-17_05-36-05'

iter_prerec = []
for dname in glob.glob('%s/prediction_*' % result_dir):
    if os.path.isdir(dname):
        for fname in glob.glob('%s/evaluation_*' % dname):
            if not os.path.exists(fname + '/pre_rec_1.npy'):
                continue

            n_iter = int(fname.split('_')[-1])
            prerecs = [n_iter]
            for ch in range(3):
                prerec = np.load(fname + '/pre_rec_%d.npy' % ch)
                prerec = np.asarray(
                    [p for p in prerec if p[0] > 0 and p[1] > 0])
                prerec = prerec[np.argmin(np.abs(prerec[:, 0] - prerec[:, 1]))]
                print ch, prerec
                prerecs.append(prerec[1])
            iter_prerec.append(prerecs)

iter_prerec = np.sort(np.asarray(iter_prerec), axis=0)

plt.plot(iter_prerec[:, 0], iter_prerec[:, 1])
plt.plot(iter_prerec[:, 0], iter_prerec[:, 2])
plt.plot(iter_prerec[:, 0], iter_prerec[:, 3])
plt.title('breakeven recall of validation set')
plt.ylabel('breakeven recall')
plt.xlabel('iteration')
plt.savefig('iter_prerec.png')
