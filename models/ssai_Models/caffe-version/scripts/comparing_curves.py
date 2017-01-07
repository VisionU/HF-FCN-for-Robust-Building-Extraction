#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import matplotlib
if 'linux' in sys.platform:
    matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import pandas as pd


def breakeven(pre_rec):
    pre_rec = np.array([[pre, rec] for pre, rec in pre_rec
                        if pre != 0 and rec != 0])
    be_pt = np.argmin(np.abs(pre_rec[:, 0] - pre_rec[:, 1]))
    pre, rec = pre_rec[be_pt]

    return pre, rec


def draw_curve(model_name, pre_rec, rec):
    plt.plot(pre_rec[:, 0], pre_rec[:, 1],
             label='%s' % (model_name))


def get_model_name_eval_dir(model, n_iter):
    model_name = re.search(
        ur'([^0-9]+)_', os.path.basename(model)).groups()[0]
    dname = '%s/prediction_%d/evaluation_%d' % (model, n_iter, n_iter)

    return model_name, dname


def compare_channel(data, lbound):
    plt.figure(data.ix['channel_in_img', 0])
    plt.ylim([lbound, 1.0])
    plt.xlim([lbound, 1.0])
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'k--')
    for i in data:
        draw_curve(data.ix['model_name', i],
                   data.ix['pre_rec', i],
                   data.ix['rec', i])
        print data.ix['model_name', i],
        print data.ix['n_iter', i]

    # Mnih's models for building
    if data.ix['channel_in_img', 0] == 1:
        plt.scatter([0.9150], [0.9150], 'x')
        plt.scatter([0.9211], [0.9211], 'x')
        plt.scatter([0.9203], [0.9203], 'x')
    plt.legend(loc='lower left')
    plt.savefig('comparing_%d.pdf' % data.ix['channel_in_img', 0],
                dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    index = ['model_name',
             'model_dir',
             'channel_in_fn',
             'channel_in_img',
             'n_iter',
             'pre',
             'rec',
             'pre_rec']
    data = pd.DataFrame(index=index)
    for model_dir in glob.glob('results/*'):
        if not os.path.isdir(model_dir):
            continue
        if 'Buildings' in model_dir:
            continue
        for eval_dir in glob.glob('%s/prediction_*' % model_dir):
            n_iter = int(re.search(ur'_([0-9]+)$', eval_dir).groups()[0])
            model_name, eval_dir = get_model_name_eval_dir(model_dir, n_iter)

            for i in range(3):
                npy_fn = '%s/pre_rec_%d.npy' % (eval_dir, i)
                if not os.path.exists(npy_fn):
                    continue
                pre_rec = np.load(npy_fn)
                pre, rec = breakeven(pre_rec)
                channel_in_fn = i
                channel_in_img = i
                if 'Building' in model_dir:
                    channel_in_fn = 0
                    channel_in_img = 1
                if 'Road' in model_dir:
                    channel_in_fn = 0
                    channel_in_img = 2
                data['%s-%d(%d)' % (model_name, channel_in_img, n_iter)] = \
                    pd.Series([model_name,
                               model_dir,
                               channel_in_fn,
                               channel_in_img,
                               n_iter,
                               pre,
                               rec,
                               pre_rec],
                              index=index)
    data.to_pickle('result.pkl')

    for i, lbound in zip(range(3), [0.975, 0.85, 0.7]):
        ch_df = data.ix[:, data.ix['channel_in_img', :] == i]
        model_dirs = ch_df.ix['model_dir', :].unique()

        models = pd.DataFrame(index=index)

        # select cols that have maximum accuracy
        # for model_dir in model_dirs:
        #     df_m = ch_df.ix[:, ch_df.ix['model_dir', :] == model_dir]
        #     df_m = df_m.ix[:, df_m.ix['rec', :].argmax()]
        #     models[df_m.ix['model_name']] = df_m

        # select max n_iter
        for model_dir in model_dirs:
            df_m = ch_df.ix[:, ch_df.ix['model_dir', :] == model_dir]
            df_m = df_m.ix[:, df_m.ix['n_iter', :].argmax()]
            models[df_m.ix['model_name']] = df_m

        models = models.T.sort('rec').T
        compare_channel(models, lbound)
