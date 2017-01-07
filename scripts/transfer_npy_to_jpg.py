import numpy as np
import cv2 as cv
from os.path import basename
import argparse
import glob

def save_npy_to_img(InputPath):
    sat_fns = np.asarray(sorted(glob.glob('%s/*.npy' % InputPath)))
    for sat_fn in sat_fns:
	sat_im = np.load(sat_fn)
	cv.imwrite(InputPath + basename(sat_fn) + '.tif',sat_im * 125)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str)
    args = parser.parse_args()
    print args

    save_npy_to_img(args.path)
