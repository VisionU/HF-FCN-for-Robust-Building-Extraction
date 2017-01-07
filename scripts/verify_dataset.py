#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lmdb
import caffe
import cv2 as cv
import numpy as np
import glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', type=str)
args = parser.parse_args()
print args

def verify_dataset(sat_db_path):
	print 'start verify'
	i = 0
	sat_env = lmdb.open(sat_db_path)
	sat_txn = sat_env.begin(write=False, buffers=False)
	sat_cur = sat_txn.cursor()
	sat_cur.next()

	for (i,(key,sat_value)) in enumerate(sat_cur):
		datum = caffe.io.caffe_pb2.Datum()
		datum.ParseFromString(sat_value)
		sat_img = caffe.io.datum_to_array(datum)
		sat_img = sat_img.swapaxes(0, 2).swapaxes(0, 1)
		print sat_img.shape
		cv.imwrite("restore"+ str(i) + ".jpg",sat_img)
		i = i + 1
		print "verify ",i
		if i > 2: 
			break


if __name__ == '__main__':
	verify_dataset(args.dataset)
