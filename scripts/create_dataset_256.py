import shutil
import os
import glob
import lmdb
import numpy as np
import cv2 as cv 
import json
import caffe
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', type=str)
args = parser.parse_args()
print args

def create_patches(sat_patch_size, map_patch_size, stride, map_ch,
                   sat_data_dir, map_data_dir,
                   sat_out_dir, map_out_dir):
    if os.path.exists(sat_out_dir):
        shutil.rmtree(sat_out_dir)
    if os.path.exists(map_out_dir):
        shutil.rmtree(map_out_dir)
    os.makedirs(sat_out_dir)
    os.makedirs(map_out_dir)

    # db
    sat_env = lmdb.Environment(sat_out_dir, map_size=1099511627776)
    sat_txn = sat_env.begin(write=True, buffers=True)
    map_env = lmdb.Environment(map_out_dir, map_size=1099511627776)
    map_txn = map_env.begin(write=True, buffers=True)

    # patch size
    sat_size = sat_patch_size
    map_size = map_patch_size
    print 'patch size:', sat_size, map_size, stride

    # get filenames
    sat_fns = np.asarray(sorted(glob.glob('%s/*.tif*' % sat_data_dir)))
    map_fns = np.asarray(sorted(glob.glob('%s/*.tif*' % map_data_dir)))
    index = np.arange(len(sat_fns))
    np.random.shuffle(index)
    sat_fns = sat_fns[index]
    map_fns = map_fns[index]

    # create keys
    keys = np.arange(15000000)
    np.random.shuffle(keys)

    n_all_files = len(sat_fns)
    print 'n_all_files:', n_all_files

    n_patches = 0
    for file_i, (sat_fn, map_fn) in enumerate(zip(sat_fns, map_fns)):
        if ((os.path.basename(sat_fn).split('.')[0])
                != (os.path.basename(map_fn).split('.')[0])):
            print 'File names are different',
            print sat_fn, map_fn
            return

        sat_im = cv.imread(sat_fn, cv.IMREAD_COLOR)
        map_im = cv.imread(map_fn, cv.IMREAD_GRAYSCALE)

        for y in range(0, sat_im.shape[0] + stride, stride):
            for x in range(0, sat_im.shape[1] + stride, stride):
                if (y + sat_size) > sat_im.shape[0]:
                    y = sat_im.shape[0] - sat_size
                if (x + sat_size) > sat_im.shape[1]:
                    x = sat_im.shape[1] - sat_size

                sat_patch = np.copy(sat_im[y:y + sat_size, x:x + sat_size])
                map_patch = np.copy(map_im[y:y + sat_size, x:x + sat_size])

                # exclude patch including big white region
                if np.sum(np.sum(sat_patch, axis=2) == (255 * 3)) > 40:
                    continue

                key = '%010d' % keys[n_patches]

                # sat db
                sat_patch = sat_patch.swapaxes(0, 2).swapaxes(1, 2)
                datum = caffe.io.array_to_datum(sat_patch, 0)
                value = datum.SerializeToString()
                sat_txn.put(key, value)

                # map db
                if map_ch == 3:
                    map_patch_multi = []
                    for ch in range(map_ch):
                        map_patch_multi.append(np.asarray(map_patch == ch,
                                                          dtype=np.uint8))
                    map_patch = np.asarray(map_patch_multi, dtype=np.uint8)
                elif map_ch == 1:
                    map_patch = map_patch.reshape((1, map_patch.shape[0],
                                                   map_patch.shape[1]))

                datum = caffe.io.array_to_datum(map_patch, 0)
                value = datum.SerializeToString()
                map_txn.put(key, value)

                n_patches += 1

                if n_patches % 10000 == 0:
                    sat_txn.commit()
                    sat_txn = sat_env.begin(write=True, buffers=True)
                    map_txn.commit()
                    map_txn = map_env.begin(write=True, buffers=True)

        print file_i, '/', n_all_files, 'n_patches:', n_patches

    sat_txn.commit()
    sat_env.close()
    map_txn.commit()
    map_env.close()
    print 'patches:\t', n_patches


if __name__ == '__main__':
         create_patches(256, 256, 64, 1,
                             args.dataset+'/mass_buildings/valid/sat',
                             args.dataset'/mass_buildings/valid/map',
                             args.dataset'/mass_buildings/lmdb/valid_sat',
                             args.dataset'/mass_buildings/lmdb/valid_map')
         create_patches(256, 256, 64, 1,
                     args.dataset'/mass_buildings/train/sat',
                     args.dataset'/mass_buildings/train/map',
                     args.dataset'/mass_buildings/lmdb/train_sat',
                     args.dataset'/mass_buildings/lmdb/train_map')
 


















