import numpy as np
import cv2 as cv
from os.path import basename
import argparse
import glob
import os

def save_npy_to_img(InputPath):
    sat_fns = np.asarray(sorted(glob.glob('%s/*.npy' % InputPath)))
    for sat_fn in sat_fns:
	sat_im = np.load(sat_fn)
	sat_im_gray = np.zeros((sat_im.shape[0],sat_im.shape[1]))
	sat_im_gray = sat_im[:,:,1]
        print sat_im_gray.shape
	cv.imwrite(InputPath + basename(sat_fn) + '.tif',sat_im_gray * 125)

def save_npy_to_buidingnpy(InputPath, BuildingPath):
    if not os.path.exists(BuildingPath):
        os.mkdir(BuildingPath)
    sat_fns = np.asarray(sorted(glob.glob('%s/*.npy' % InputPath)))
    for sat_fn in sat_fns:
	sat_im = np.load(sat_fn)
	sat_im_building = np.zeros((sat_im.shape[0],sat_im.shape[1],1))
	sat_im_building[:,:,0] = sat_im[:,:,1]
        print sat_im_building.shape
	np.save(BuildingPath + '/pred_0_' + basename(sat_fn),sat_im_building)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str)
    parser.add_argument('--bldpath','-bldp', type=str)
    args = parser.parse_args()
    print args

    #save_npy_to_img(args.path)
    save_npy_to_buidingnpy(args.path, args.bldpath)
