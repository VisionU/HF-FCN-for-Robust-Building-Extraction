#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import glob
import subprocess
import argparse
import os
from multiprocessing import Process
from os.path import basename
from collections import defaultdict

get_sat_dir = lambda t: '../../../data/mass_%s/test/sat' % t.lower()
get_map_dir = lambda t: '../../../data/mass_%s/test/map' % t.lower()
get_iter = lambda fn: int(re.search(ur'_([0-9]+)\.', basename(fn)).groups()[0])

channel = None
script_dir = '../../../scripts'
result_dir = '.'


def parallel_pred_eval(pred, evaluate, model_dir, n_iter, snapshot_fn,
                       test_sat_dir, test_map_dir, channel, offset):
    worker = Process(
        target=pred, args=(model_dir, snapshot_fn, test_sat_dir, channel, offset))
    worker.start()
    worker.join()

    result_dir = 'prediction_%d' % n_iter
    worker = Process(target=evaluate,
                     args=(model_dir, test_map_dir, result_dir, channel, offset))
    worker.start()
    worker.join()


def predict(model_dir, snapshot_fn, test_sat_dir, channel, offset):
    os.chdir(model_dir)
    print subprocess.check_output(['ls'])
    cmd = [
        'python', '%s/test_prediction.py' % script_dir,
        '--model', 'predict.prototxt',
        '--weight', 'snapshots/%s' % basename(snapshot_fn),
        '--img_dir', test_sat_dir,
        '--channel', str(channel),
        '--offset', str(offset)
    ]
    subprocess.check_output(cmd)


def evaluate(model_dir, test_map_dir, result_dir, channel, offset):
    os.chdir(model_dir)
    print test_map_dir, os.path.exists(test_map_dir)
    cmd = [
        'python', '%s/test_evaluation.py' % script_dir,
        '--map_dir', test_map_dir,
        '--result_dir', result_dir,
        '--channel', str(channel),
        '--offset', str(offset)
    ]
    subprocess.check_output(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--offset', '-o', type=bool, default=False)
    args = parser.parse_args()
    print args

    offset = defaultdict(int)
    for model_dir in glob.glob('%s/*' % result_dir):
        print model_dir
        for snapshot_fn in glob.glob('%s/snapshots/*.caffemodel' % model_dir):
            n_iter = get_iter(snapshot_fn)
            if n_iter % 100000 == 0:
                dname = os.path.dirname(snapshot_fn)
                pred_dname = '%s/prediction_%d' % (model_dir, n_iter)

                if 'Buildings_2015' in model_dir:
                    channel = 1
                    test_sat_dir = get_sat_dir('buildings')
                    test_map_dir = get_map_dir('buildings')
                elif 'Roads_2015' in model_dir:
                    channel = 1
                    test_sat_dir = get_sat_dir('roads')
                    test_map_dir = get_map_dir('roads')
                elif 'Roads-Mini_2015' in model_dir:
                    channel = 1
                    test_sat_dir = get_sat_dir('roads_mini')
                    test_map_dir = get_map_dir('roads_mini')
                else:
                    channel = 3
                    test_sat_dir = get_sat_dir('merged')
                    test_map_dir = get_map_dir('merged')

                worker = Process(target=parallel_pred_eval,
                                 args=(predict, evaluate, model_dir, n_iter, snapshot_fn,
                                       test_sat_dir, test_map_dir, channel, offset[n_iter]))
                worker.start()

                if args.offset:
                    offset[n_iter] += 1

                print pred_dname, channel
