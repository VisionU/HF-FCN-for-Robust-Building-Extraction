#!/bin/bash
python  ../scripts/run_evaluation_complex.py  \
	--gt_dir ../results/GroundTruth   \
	--pred_dir ../results/whole_image_results/Mnih-Machine-PHDthesis13/Mnih \
	--pad 24 \
	--relax 0 \
	--pr_dir ../results/whole_image_results/PresicionRecallComparision

python  ../scripts/run_evaluation_complex.py  \
	--gt_dir ../results/GroundTruth   \
	--pred_dir ../results/whole_image_results/Mnih-Machine-PHDthesis13/Mnih \
	--pad 24 \
	--relax 3 \
	--pr_dir ../results/whole_image_results/PresicionRecallComparision

python  ../scripts/run_evaluation_complex.py  \
	--gt_dir ../results/GroundTruth   \
	--pred_dir ../results/whole_image_results/Mnih-Machine-PHDthesis13/Mnih-CRF \
	--pad 24 \
	--relax 0 \
	--pr_dir ../results/whole_image_results/PresicionRecallComparision

python  ../scripts/run_evaluation_complex.py  \
	--gt_dir ../results/GroundTruth   \
	--pred_dir ../results/whole_image_results/Mnih-Machine-PHDthesis13/Mnih-CRF \
	--pad 24 \
	--relax 3 \
	--pr_dir ../results/whole_image_results/PresicionRecallComparision

python  ../scripts/run_evaluation_complex.py  \
	--gt_dir ../results/GroundTruth   \
	--pred_dir ../results/whole_image_results/Saito-Multiple-JIST16/Saito-MA \
	--pad 31 \
	--relax 0 \
	--pr_dir ../results/whole_image_results/PresicionRecallComparision


python  ../scripts/run_evaluation_complex.py  \
	--gt_dir ../results/GroundTruth   \
	--pred_dir ../results/whole_image_results/Saito-Multiple-JIST16/Saito-MA\
	--pad 31 \
	--relax 3 \
	--pr_dir ../results/whole_image_results/PresicionRecallComparision

python  ../scripts/run_evaluation_complex.py  \
	--gt_dir ../results/GroundTruth   \
	--pred_dir ../results/whole_image_results/Saito-Multiple-JIST16/Saito-MA-CIS \
	--pad 31 \
	--relax 0 \
	--pr_dir ../results/whole_image_results/PresicionRecallComparision

python  ../scripts/run_evaluation_complex.py  \
	--gt_dir ../results/GroundTruth   \
	--pred_dir ../results/whole_image_results/Saito-Multiple-JIST16/Saito-MA-CIS  \
	--pad 31 \
	--relax 3 \
	--pr_dir ../results/whole_image_results/PresicionRecallComparision


python  ../scripts/run_evaluation_complex.py  \
	--gt_dir ../results/GroundTruth   \
	--pred_dir ../results/whole_image_results/Zuo-HF-FCN-ACCV16  \
	--pad 0 \
	--relax 0 \
	--pr_dir ../results/whole_image_results/PresicionRecallComparision



python  ../scripts/run_evaluation_complex.py  \
	--gt_dir ../results/GroundTruth   \
	--pred_dir ../results/whole_image_results/Zuo-HF-FCN-ACCV16  \
	--pad 0 \
	--relax 3 \
	--pr_dir ../results/whole_image_results/PresicionRecallComparision

python ../scripts/merge_pr_curve_bpts.py \
	--pr_curve_bpts_dir ../results/whole_image_results/PresicionRecallComparision






















