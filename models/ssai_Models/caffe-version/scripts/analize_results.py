#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import glob
import os

fp_merged = []
fp_building = []
fp_road = []

result_dir = 'Mnih_CNN_Buildings'

for dname in sorted(glob.glob('%s/*' % result_dir)):
    if not os.path.isdir(dname):
        continue
    model_name = os.path.basename(dname)
    solver_fn = '%s/solver.prototxt' % dname
    print model_name,
    if not os.path.exists(solver_fn):
        continue
    print dname

    base_lr = None
    gamma = None
    stepsize = None
    momentum = None
    weight_decay = None
    for line in open(solver_fn):
        if 'base_lr' in line:
            base_lr = float(line.split(':')[-1].strip())
        if 'gamma' in line:
            gamma = float(line.split(':')[-1].strip())
        if 'stepsize' in line:
            stepsize = int(line.split(':')[-1].strip())
        if 'momentum' in line:
            momentum = float(line.split(':')[-1].strip())
        if 'weight_decay' in line:
            weight_decay = float(line.split(':')[-1].strip())

    for pred_dir in sorted(glob.glob('%s/prediction_*' % dname)):
        iter = int(pred_dir.split('_')[-1].strip())
        print iter
        recalls = []
        for pre_rec_fn in sorted(glob.glob(
                '%s/evaluation_%d/*.npy' % (pred_dir, iter))):
            pre_rec = np.load(pre_rec_fn)
            pre_rec = np.array([[pre, rec] for pre, rec in pre_rec
                                if pre != 0 and rec != 0])
            be_pt = np.abs(pre_rec[:, 0] - pre_rec[:, 1]).argmin()
            ch = int(pre_rec_fn.split('_')[-1].split('.')[0])
            pre, rec = pre_rec[be_pt]
            recalls.append(rec)

        print 'len(recalls):', len(recalls)
        if len(recalls) == 0:
            continue

        msg = model_name + ','
        msg += str(iter) + ','
        msg += ','.join([str(r) for r in recalls])
        if len(recalls) == 3:
            msg += ',' + str(np.mean(recalls[1:]))
        msg += ',' + str(base_lr)
        msg += ',' + str(gamma)
        msg += ',' + str(stepsize)
        msg += ',' + str(momentum)
        msg += ',' + str(weight_decay)

        if len(recalls) == 3:
            fp_merged.append([np.mean(recalls[1:]), msg])

        if len(recalls) == 1 and 'Buildings' in model_name:
            fp_building.append([recalls[0], msg])

        if len(recalls) == 1 and 'Roads' in model_name:
            fp_road.append([recalls[0], msg])

# fp_merged = np.array(fp_merged)
fp_building = np.array(fp_building)
# fp_road = np.array(fp_road)

# fp_merged = fp_merged[np.argsort(fp_merged[:, 0])]
fp_building = fp_building[np.argsort(fp_building[:, 0])]
# fp_road = fp_road[np.argsort(fp_road[:, 0])]

fp = open('result.csv', 'w')

# print >> fp, 'model,iteration,other,building,road,average,base_lr,gamma,stepsize,momentum,weight_decay'
# for line in fp_merged[:, 1]:
#     print >> fp, line

print >> fp, ''
print >> fp, 'model,iteration,bulding,base_lr,gamma,stepsize,momentum,weight_decay'
for line in fp_building[:, 1]:
    print >> fp, line
#
# print >> fp, ''
# print >> fp, 'model,iteration,road,base_lr,gamma,stepsize,momentum,weight_decay'
# for line in fp_road[:, 1]:
#     print >> fp, line
