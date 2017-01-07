#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import lmdb
import caffe
import cv2 as cv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--db_type', '-t', type=str)
parser.add_argument('--sat_db_path', '-s', type=str)
parser.add_argument('--map_db_path', '-m', type=str)
args = parser.parse_args()
print args

out_dir = 'test_%s' % args.db_type
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

sat_db_path = args.sat_db_path
map_db_path = args.map_db_path

# sat db
sat_env = lmdb.open(sat_db_path)
sat_txn = sat_env.begin(write=False, buffers=False)
sat_cur = sat_txn.cursor()

# map db
map_env = lmdb.open(map_db_path)
map_txn = map_env.begin(write=False, buffers=False)
map_cur = map_txn.cursor()

sat_cur.next()
map_cur.next()

for i in range(100):
    sat_key, sat_value = sat_cur.item()
    map_key, map_value = map_cur.item()

    datum = caffe.io.caffe_pb2.Datum()
    datum.ParseFromString(sat_value)
    sat_img = caffe.io.datum_to_array(datum)

    datum = caffe.io.caffe_pb2.Datum()
    datum.ParseFromString(map_value)
    map_img = caffe.io.datum_to_array(datum)

    print map_img

    sat_img = sat_img.transpose([1, 2, 0])
    map_img = map_img.transpose([1, 2, 0])

    map_canvas = np.copy(sat_img)
    h, w, _ = map_canvas.shape
    mh, mw, _ = map_img.shape
    map_canvas[h / 2 - mh / 2:h / 2 + mh / 2,
               w / 2 - mw / 2:w / 2 + mw / 2] = map_img * 255

    out_img = np.hstack((sat_img, map_canvas))

    cv.imwrite('%s/%03d.png' % (out_dir, i), out_img)

    sat_cur.next()
    map_cur.next()
