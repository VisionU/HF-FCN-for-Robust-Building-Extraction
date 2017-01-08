import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import numpy as np
import ctypes
import os
import re 
from os.path import basename
from os.path import exists
from multiprocessing import Queue, Process, Array
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pr_curve_bpts_dir','-prd',type=str)
args = parser.parse_args()
print args

def draw_pre_rec_curve(pre_rec, lim_x,lim_y):
	plt.clf()
	legend = ['Mnih-CNN[12]','Mnih-CNN+CRF[12]','Saito-multi-MA[13]','Saito-multi-MA&CIS[13]','Ours']
	for i in range(pre_rec.shape[0]):
		plt.plot(pre_rec[i,:, 0], pre_rec[i,:, 1], label = legend[i])
		# plt.plot(breakeven_pt[i,0], breakeven_pt[i,1],
		# 	 'x', label='breakeven recall: %f' % (breakeven_pt[i,1]))
	plt.ylabel('recall')
	plt.xlabel('precision')
	plt.ylim([lim_x, lim_y])
	plt.xlim([lim_x, lim_y])
	plt.legend(loc='lower left')
	plt.grid(linestyle='--')


def get_pre_recall_breakeven(rootdir):

	files = sorted(glob.glob('%s/*.npy' % rootdir))

	num = len(files) / 4
	relax0_pre_recall = np.zeros((num,256,2),dtype=float)
	relax3_pre_recall = np.zeros((num,256,2),dtype = float)
	relax0_breakeven = np.zeros((num,2),dtype = float)
	relax3_breakeven = np.zeros((num,2),dtype = float)

	i = 0
	j = 0
	k = 0
	m = 0
	for file in files:
		npyfile = np.load(file)
		if 'pre_rec_ch0_relax_0' in file:
			relax0_pre_recall[i,:] = npyfile
			print file
			i+=1
		elif 'pre_rec_ch0_relax_3' in file:
			relax3_pre_recall[j,:] = npyfile
			print file
			j+=1
		elif 'breakeven_pt_ch0_relax_0' in file:
			relax0_breakeven[k,:] = npyfile
			k+=1
		elif 'breakeven_pt_ch0_relax_3' in file:
			relax3_breakeven[m,:] = npyfile
			m+=1
	return  relax0_pre_recall, relax3_pre_recall, relax0_breakeven, relax3_breakeven

if __name__ == '__main__':
	relax0_pre_recall, relax3_pre_recall, relax0_breakeven,relax3_breakeven = get_pre_recall_breakeven(args.pr_curve_bpts_dir)

	fig_r3 = plt.figure()
	title = 'Relexed precision-recal curve (relax = 3)'
	draw_pre_rec_curve(relax3_pre_recall,0.86,1)
	fig_r3.savefig(args.pr_curve_bpts_dir +'/merged_pr_curve_relax3.tif')

	fig_r0 = plt.figure()
	title = 'Relexed precision-recal curve (relax = 0)'
	draw_pre_rec_curve(relax0_pre_recall,0.55,1)
	fig_r0.savefig(args.pr_curve_bpts_dir + '/merged_pr_curve_relax0.tif')
