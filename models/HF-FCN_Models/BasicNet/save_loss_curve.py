#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import numpy as np
if sys.platform.startswith('linux'):
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start', '-s', type=int, default=0)
args = parser.parse_args()
print args


def save_loss_curve(dname, fname='caffe.INFO'):
    l_iter = []
    loss = []
    t_iter = []
    error = []
    test = []

    if os.path.exists('%s/nohup.out' % dname):
        fname = 'nohup.out'
    for line in open('%s/%s' % (dname, fname)):
        if 'Iteration' in line and 'loss' in line:
            tmp = re.search(ur'Iteration\s([0-9]+),', line)
            if tmp:
                l_iter.append(int(tmp.groups()[0]))
        if 'Train net output' in line and 'predict_loss' in line:
            tmp = re.search(ur'=\s([0-9\.]+)\sloss', line)
            if tmp:
                loss.append(float(tmp.groups()[0]))
        if 'Iteration' in line and 'Testing net' in line:
            tmp = re.search(ur'Iteration\s([0-9]+),', line)
            if tmp:
                t_iter.append(int(tmp.groups()[0]))
        if 'Test net output' in line and 'error_rate = ' in line:
            tmp = re.search(ur'=\s([0-9\.]+)\sloss', line)
            if tmp:
                error.append(float(tmp.groups()[0]))
        if 'Test net output' in line and 'predict_loss = ' in line:
            tmp = re.search(ur'=\s([0-9\.]+)\sloss', line)
            if tmp:
                test.append(float(tmp.groups()[0]))

    l_iter = l_iter[args.start:]
    loss = loss[args.start:]
    t_iter = t_iter[args.start:]
    error = error[args.start:]
    test = test[args.start:]
    print t_iter
    print test

    plt.clf()
    min_t = 0.0 if len(loss) == 0 else np.min(loss)
    pos_t = 0 if len(loss) == 0 else l_iter[np.argmin(loss)]
    min_l = 0.0 if len(test) == 0 else np.min(test)
    pos_l = 0 if len(test) == 0 else l_iter[np.argmin(test)]
    min_e = 0.0 if len(error) == 0 else np.min(error)
    pos_e = 0 if len(error) == 0 else t_iter[np.argmin(error)]

    plt.subplot(2, 1, 1)
    title = '%s\nmin training loss:%f (%d)\n' % (
        os.path.basename(dname), min_t, pos_t)
    title += 'min valid loss:%f (%d)\n' % (min_l, pos_l)
    title += 'min error: %f (%d)' % (min_e, pos_e)
    plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.plot(l_iter[:len(loss)], loss, label='traning loss')
    plt.plot(t_iter[:len(test)], test, label='test loss')
    plt.legend(loc='upper right')
    plt.grid(linestyle='--')

    plt.subplot(2, 1, 2)
    plt.xlabel('iteration')
    plt.ylabel('error')
    plt.plot(t_iter[:len(error)], error, label='error rate')
    plt.legend(loc='upper right')
    plt.grid(linestyle='--')

    plt.savefig('%s/loss.png' % dname, bbox_inches='tight')


if __name__ == '__main__':
    save_loss_curve('./', 'nohup.out')
