#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import matplotlib
matplotlib.use('Agg')
sys.path.append('ssai-lib/build')
sys.path.append('../ssai-lib/build')
sys.path.append('../../ssai-lib/build')
import matplotlib.pyplot as plt
from ssai import relax_precision, relax_recall
import cv2 as cv
import glob
import numpy as np
import ctypes
import os
from os.path import basename
from os.path import exists
from multiprocessing import Queue, Process, Array
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', '-g', type=str)
parser.add_argument('--pred_dir', '-d', type=str)
parser.add_argument('--pr_dir','-p',type=str)
parser.add_argument('--channel', '-c', type=int, default=1)
parser.add_argument('--offset', '-o', type=int, default=0)
parser.add_argument('--pad', '-a', type=int, default=0)  # (64 / 2) - (16 / 2)
parser.add_argument('--relax', '-r', type=int, default=0)  
args = parser.parse_args()
print args

ch = args.channel
steps = 256
relax = args.relax
pad = args.pad
n_thread = 8
result_dir = args.pred_dir
label_dir = args.gt_dir
pr_curve_bpts_dir = args.pr_dir

result_fns = sorted(glob.glob('%s/*.npy' % result_dir))
n_results = len(result_fns)
eval_dir = '%s/evaluation_relax%d' % (result_dir, relax)
print eval_dir
print pr_curve_bpts_dir

all_positive_base = Array(ctypes.c_double, n_results * ch * steps)
all_positive = np.ctypeslib.as_array(all_positive_base.get_obj())
all_positive = all_positive.reshape((n_results, ch, steps))

all_prec_tp_base = Array(ctypes.c_double, n_results * ch * steps)
all_prec_tp = np.ctypeslib.as_array(all_prec_tp_base.get_obj())
all_prec_tp = all_prec_tp.reshape((n_results, ch, steps))

all_true_base = Array(ctypes.c_double, n_results * ch * steps)
all_true = np.ctypeslib.as_array(all_true_base.get_obj())
all_true = all_true.reshape((n_results, ch, steps))

all_recall_tp_base = Array(ctypes.c_double, n_results * ch * steps)
all_recall_tp = np.ctypeslib.as_array(all_recall_tp_base.get_obj())
all_recall_tp = all_recall_tp.reshape((n_results, ch, steps))


def makedirs(dname):
    if not exists(dname):
        os.makedirs(dname)


def get_pre_rec(positive, prec_tp, true, recall_tp, steps):
    pre_rec = []
    breakeven = []
    for t in range(steps):
        if positive[t] < prec_tp[t] or true[t] < recall_tp[t]:
            sys.exit('calculation is wrong')
        pre = float(prec_tp[t]) / positive[t] if positive[t] > 0 else 0
        rec = float(recall_tp[t]) / true[t] if true[t] > 0 else 0
        pre_rec.append([pre, rec])
        if pre != 1 and rec != 1 and pre > 0 and rec > 0:
            breakeven.append([pre, rec])
    pre_rec = np.asarray(pre_rec)
    breakeven = np.asarray(breakeven)
    breakeven_pt = np.abs(breakeven[:, 0] - breakeven[:, 1]).argmin()
    breakeven_pt = breakeven[breakeven_pt]

    return pre_rec, breakeven_pt


def draw_pre_rec_curve(pre_rec, breakeven_pt):
    plt.clf()
    plt.plot(pre_rec[:, 0], pre_rec[:, 1])
    plt.plot(breakeven_pt[0], breakeven_pt[1],
             'x', label='breakeven recall: %f' % (breakeven_pt[1]))
    plt.ylabel('recall')
    plt.xlabel('precision')
    plt.ylim([0.0, 1.1])
    plt.xlim([0.0, 1.1])
    plt.legend(loc='lower left')
    plt.grid(linestyle='--')


def worker_thread(result_fn_queue):
    while True:
        i, result_fn = result_fn_queue.get()
        if result_fn is None:
            break

        img_id = basename(result_fn).split('pred_')[-1]
        img_id, _ = os.path.splitext(img_id)
        if '.' in img_id:
            img_id = img_id.split('.')[0]
        if len(re.findall(ur'_', img_id)) > 1:
            img_id = '_'.join(img_id.split('_')[1:])
        out_dir = '%s/%s' % (eval_dir, img_id)
        makedirs(out_dir)
        print img_id

	
        label = cv.imread('%s/%s.tif' %
                          (label_dir, img_id), cv.IMREAD_GRAYSCALE)

	pred = np.load(result_fn)
       
        
        label = label[pad + args.offset:pad + args.offset + pred.shape[0],
                      pad + args.offset:pad + args.offset + pred.shape[1]]
        cv.imwrite('%s/label_%s.png' % (out_dir, img_id), label * 125)

        for c in range(ch):
            for t in range(0, steps):
                threshold = 1.0 / steps * t

                pred_vals = np.array(
                    pred[:, :, c] >= threshold, dtype=np.int32)

                label_vals = np.array(label, dtype=np.int32)
                if ch > 1:
                    label_vals = np.array(label == c, dtype=np.int32)

                all_positive[i, c, t] = np.sum(pred_vals)
                all_prec_tp[i, c, t] = relax_precision(
                    pred_vals, label_vals, relax)

                all_true[i, c, t] = np.sum(label_vals)
                all_recall_tp[i, c, t] = relax_recall(
                    pred_vals, label_vals, relax)

            pre_rec, breakeven_pt = get_pre_rec(
                all_positive[i, c], all_prec_tp[i, c],
                all_true[i, c], all_recall_tp[i, c], steps)

            draw_pre_rec_curve(pre_rec, breakeven_pt)
            plt.savefig('%s/pr_curve_%d.png' % (out_dir, c))
            np.save('%s/pre_rec_%d' % (out_dir, c), pre_rec)
            cv.imwrite('%s/pred_%d.png' % (out_dir, c), pred[:, :, c] * 255)

            print img_id, c, breakeven_pt
    print 'thread finished'


if __name__ == '__main__':
    makedirs(pr_curve_bpts_dir)
    result_fn_queue = Queue()
    workers = [Process(target=worker_thread,
                       args=(result_fn_queue,)) for i in range(n_thread)]
    map(lambda w: w.start(), workers)
    [result_fn_queue.put((i, fn)) for i, fn in enumerate(result_fns)]
    [result_fn_queue.put((None, None)) for _ in range(n_thread)]
    map(lambda w: w.join(), workers)
    print 'all finished'

    all_positive = np.sum(all_positive, axis=0)
    all_prec_tp = np.sum(all_prec_tp, axis=0)
    all_true = np.sum(all_true, axis=0)
    all_recall_tp = np.sum(all_recall_tp, axis=0)
    for c in range(ch):
        pre_rec, breakeven_pt = get_pre_rec(
            all_positive[c], all_prec_tp[c],
            all_true[c], all_recall_tp[c], steps)
        draw_pre_rec_curve(pre_rec, breakeven_pt)
        plt.savefig('%s/%s_pr_curve_ch%d_relax%d.png' % (pr_curve_bpts_dir, basename(result_dir),c,relax))
        np.save('%s/%s_pre_rec_ch%d_relax_%d' % (pr_curve_bpts_dir, basename(result_dir), c,relax), pre_rec)
	np.save('%s/%s_breakeven_pt_ch%d_relax_%d' % (pr_curve_bpts_dir,basename(result_dir), c,relax), breakeven_pt)
        print breakeven_pt

