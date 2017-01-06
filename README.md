This is a state-of-the-art project for building extraction in high resolution remote sensing image using dataset [Massachusetts road & building dataset](https://www.cs.toronto.edu/~vmnih/data/) . And, our approach was published in ACCV 2016, clik here to download [our paper](https://github.com/tczuo/HF-FCN-for-Robust-Building-Extraction/blob/master/0663.pdf)

# Requirements
- Last version caffe
- OpenCV 3.0.0
- NumPy
- CUDA V8.0
- CUDNN V5.0

# Create Dataset
$ sh shells/download_minh_dataset.sh  <br />
$ python scripts/create_dataset_256.py  <br />
$ python scripts/verify_dataset.py -d /data/mass_building/lmdb/train/  <br />
  
# Start Training
$ cd models/HF-FCN_Models/BasicNet/  <br />
$ nohup python solve.py&  <br />

# Prediction
$ cd results/
$ python ../scripts/test_prediction.py --model ../models/predict.prototxt --weight ../modles/snapshots/BuildingDetection_iter_12000.caffemodel --img_dir /data/mass_buildings/test/sat

# Evaluation
$ cd results/prediction_12000
$ python ../../scripts/test_evaluation.py --map_dir /data/mass_buildings/test/map --result_dir prediction_12000 

# Results Display
    \begin{table} 
    \centering
	\caption{Performance comparison with \cite{Mnih2013Machine,Saito2016Multiple}. Recall here  means recall at breakeven points. Time is computed in the same computer with a single NVIDIA Titan 12GB GPU.}
	\begin{tabular}{L{38mm}C{26mm}C{26mm}C{26mm}}     
	\toprule
	& Recall ($\rho$ = 3) & Recall ($\rho$ = 0)& Time (s)\\
	\midrule
	Mnih-CNN \cite{Mnih2013Machine} & 0.9271 & 0.7661 & 8.70  \\ 
	Mnih-CNN+CRF \cite{Mnih2013Machine} & 0.9282 & 0.7638 & 26.60\\ 
	Saito-multi-MA \cite{Saito2016Multiple} & 0.9503 & 0.7873 & 67.72 \\
	Saito-multi-MA$\&$CIS \cite{Saito2016Multiple} & 0.9509 & 0.7872 & 67.84 \\
	Ours (HF-FCN) & $\bm{0.9643}$ & $\bm{0.8424}$ & $\bm{1.07}$\\
	\bottomrule
	\end{tabular}
	\label{tab:PerformanceComparision}
	\end{table}  

# Pre-trained models and Predicted results


# Reference
If you use this code for your project, please cite this conference paper:
Tongchun Zuo, Juntao Feng, Xuejin Chen. "HF-FCN: Hierarchically Fused Fully Convolutional Network for Robust Building Extraction". Asian Conference of Computer Vision. 2016. 
