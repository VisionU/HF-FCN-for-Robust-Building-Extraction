import matplotlib.pyplot as plt
import cv2 as cv
import glob
import numpy as np
import argparse
import os
from os.path import exists

parser = argparse.ArgumentParser()
parser.add_argument('--AerialImageDir', '-A', type=str)
parser.add_argument('--GroundTruthDir', '-G', type=str)
parser.add_argument('--PredictedImageDir','-P',type=str)
parser.add_argument('--width', '-w', type=int, default=256)
parser.add_argument('--offset', '-o', type=int, default=0)
parser.add_argument('--ResultDir', '-R', type=str)  # (64 / 2) - (16 / 2)
parser.add_argument('--AerialPatchesDir','-e',type=str)
parser.add_argument('--PatchesGroundTruthDir','-g',type=str)
args = parser.parse_args()
print args

def create_show_for_patch(AerialImagefns,GroundTruthfns,PredictedImagefns,ResultDir,AerialPatchesDir,PatchesGroundTruthDir,crop_id_pos,width,offset):
	for num in range(0,len(crop_id_pos) / 3):
		print num
		AerialImage = cv.imread(AerialImagefns[crop_id_pos[num * 3]])
		GroundTruth =  cv.imread(GroundTruthfns[crop_id_pos[num * 3]],cv.IMREAD_GRAYSCALE)
		PredictedImage = np.load(PredictedImagefns[crop_id_pos[num * 3]])
		AerialPatch = AerialImage[crop_id_pos[num*3+1]:crop_id_pos[num*3+1]+width,crop_id_pos[num*3+2]:crop_id_pos[num*3+2]+width,:]
		GTPatch = GroundTruth[crop_id_pos[num*3+1]:crop_id_pos[num*3+1] + width,crop_id_pos[num*3+2] :crop_id_pos[num*3+2] + width]
		#PredictedImagePatch = np.zeros((width,width,1))
		PredictedImagePatch = PredictedImage[crop_id_pos[num*3+1]- offset:crop_id_pos[num*3+1]- offset + width,crop_id_pos[num*3+2]- offset:crop_id_pos[num*3+2]- offset +width]

		# save aerialpatch and gtpatch for evaluation
		
		cv.imwrite(AerialPatchesDir + '/AerialPatch' + str(num) + '.tif',AerialPatch)
		np.save(ResultDir + '/pred_' + str(num),PredictedImagePatch)
		cv.imwrite(ResultDir + '/pred_' + str(num) + '.tif',PredictedImagePatch * 255)
		cv.imwrite(PatchesGroundTruthDir +'/' + str(num) + '.tif',GTPatch)

		#create show
		for r in range(AerialPatch.shape[0]):
			for c in range(AerialPatch.shape[1]):
				if (GTPatch[r,c] > 0.8) and (PredictedImagePatch[r,c] >= 0.5):
					AerialPatch[r,c,:] = (0,200,0)
				elif (GTPatch[r,c] < 0.2) and (PredictedImagePatch[r,c] >= 0.5):
					AerialPatch[r,c,:] = (255,0,0)
				elif (GTPatch[r,c] > 0.8) and (PredictedImagePatch[r,c] < 0.5):
					AerialPatch[r,c,:] = (0,0,200)
				# else:
				# 	AerialPatch[r,c,:] += 60
		cv.imwrite(ResultDir + '/IllustratedPatch' + str(num) + '.tif',AerialPatch)
def makedirs(dname):
    if not exists(dname):
        os.makedirs(dname)


if __name__ == '__main__':

	AerialImagefns = sorted(glob.glob('%s/*.tiff' % args.AerialImageDir))
	GroundTruthfns = sorted(glob.glob('%s/*.tif' % args.GroundTruthDir))
	PredictedImagefns = sorted(glob.glob('%s/*.npy' % args.PredictedImageDir))
	crop_id_pos = [0,651,1091,3,590,1196,6,144,812,6,1100,31,9,38,60,4,580,606,6,175,491]
	#crop_id_pos = [4,0,0,6,0,0]
	width = args.width
	offset = args.offset
	ResultDir = args.ResultDir
	AerialPatchesDir = args.AerialPatchesDir
	PatchesGroundTruthDir = args.PatchesGroundTruthDir
	makedirs(ResultDir)
	makedirs(AerialPatchesDir)
	makedirs(PatchesGroundTruthDir)
	create_show_for_patch(AerialImagefns,GroundTruthfns,PredictedImagefns,ResultDir,AerialPatchesDir,PatchesGroundTruthDir,crop_id_pos,width,offset)

