#!/bin/bash
python  ../scripts/select_patches_illustration.py \
	--AerialImageDir ../results/TestAerialImages   \
	--GroundTruthDir ../results/GroundTruth \
	--PredictedImageDir ../results/whole_image_results/Mnih-Machine-PHDthesis13/Mnih-CRF \
	--width 256 \
	--offset 24 \
	--ResultDir ../results/challenging_region_results/Mnih-CRF \
	--AerialPatchesDir ../results/challenging_region_results/AerialPatches \
	--PatchesGroundTruthDir  ../results/challenging_region_results/PatchesGroundTruth

python  ../scripts/select_patches_illustration.py \
	--AerialImageDir ../results/TestAerialImages   \
	--GroundTruthDir ../results/GroundTruth \
	--PredictedImageDir ../results/whole_image_results/Saito-Multiple-JIST16/Saito-MA-CIS \
	--width 256 \
	--offset 31 \
	--ResultDir ../results/challenging_region_results/Saito-MA-CIS \
	--AerialPatchesDir ../results/challenging_region_results/AerialPatches \
	--PatchesGroundTruthDir  ../results/challenging_region_results/PatchesGroundTruth

python  ../scripts/select_patches_illustration.py \
	--AerialImageDir ../results/TestAerialImages   \
	--GroundTruthDir ../results/GroundTruth \
	--PredictedImageDir ../results/whole_image_results/Zuo-HF-FCN-ACCV16 \
	--width 256 \
	--offset 0 \
	--ResultDir ../results/challenging_region_results/Zuo-HF-FCN-ACCV16 \
	--AerialPatchesDir ../results/challenging_region_results/AerialPatches \
	--PatchesGroundTruthDir  ../results/challenging_region_results/PatchesGroundTruth

python  ../scripts/run_evaluation_complex.py  \
	--gt_dir ../results/challenging_region_results/PatchesGroundTruth   \
	--pred_dir ../results/challenging_region_results/Mnih-CRF \
	--pad 0 \
	--relax 0 \
	--pr_dir ../results/challenging_region_results/PresicionRecallComparision

python  ../scripts/run_evaluation_complex.py  \
	--gt_dir ../results/challenging_region_results/PatchesGroundTruth   \
	--pred_dir ../results/challenging_region_results/Saito-MA-CIS  \
	--pad 0 \
	--relax 0 \
	--pr_dir ../results/challenging_region_results/PresicionRecallComparision


python  ../scripts/run_evaluation_complex.py  \
	--gt_dir ../results/challenging_region_results/PatchesGroundTruth   \
	--pred_dir ../results/challenging_region_results/Zuo-HF-FCN-ACCV16  \
	--pad 0 \
	--relax 0 \
	--pr_dir ../results/challenging_region_results/PresicionRecallComparision


















