This is a state-of-the-art project for building extraction in high resolution remote sensing image using dataset [Massachusetts road & building dataset](https://www.cs.toronto.edu/~vmnih/data/) . And, our approach was published in ACCV 2016, clik here to download [our paper](https://github.com/tczuo/HF-FCN-for-Robust-Building-Extraction/blob/master/0663.pdf)

# Requirements
- Last version caffe
- OpenCV 3.1.0
- NumPy
- Boost 1.59.0
- CUDA V8.0
- CUDNN V5.0

# Create Dataset
$ sh shells/download_minh_dataset.sh  <br />
$ python scripts/create_dataset_256.py  <br />
$ python scripts/verify_dataset.py -d /data/mass_building/lmdb/train_sat_256  <br />
  
# Start Training
$ cd models/HF-FCN_Models/BasicNet/  <br />
$ nohup python solve.py&  <br />

# Prediction
$ cd results/   <br />
$ python ../scripts/run_prediction.py   <br />
 \qquad  --model ../models/HF-FCN_Models/BasicNet/predict.prototxt   <br />
 \qquad	 --weight ../weights/HF-FCN_iter_12000.caffemodel   <br />
 \qquad	 --img_dir /data/mass_buildings/source/test/sat   <br />

# Evaluation
$ cd results/prediction_12000   <br />
$ python ../../scripts/test_evaluation.py   <br />
\qquad	--map_dir /data/mass_buildings/test/map   <br />
\qquad	--result_dir prediction_12000   <br />

# Results Display
|                                                | Recall ( \rho = 3) | Recall ( \rho = 0) | Time (s) |
|------------------------------------------------|---------------------|---------------------|----------|
| Mnih-CNN \cite{Mnih2013Machine}                | 0.9271              | 0.7661              | 8.70     |
| Mnih-CNN+CRF \cite{Mnih2013Machine}            | 0.9282              | 0.7638              | 26.60    |
| Saito-multi-MA \cite{Saito2016Multiple}        | 0.9503              | 0.7873              | 67.72    |
| Saito-multi-MA&CIS \cite{Saito2016Multiple} | 0.9509              | 0.7872              | 67.84    |
| Ours (HF-FCN)                                  | 0.9643              | 0.8424              |   1.07   |


# Pre-trained models
HF-FCN16-iter-12000.caffemodel
Minh13-Machine.caffemodel
[Saito16-Multiple-caffemodels](https://github.com/mitmul/ssai-cnn/wiki/Pre-trained-models)

# Predicted results
HF-FCN16-results
Mnih13-Machine-results
[Saito16-Multiple-results](https://github.com/mitmul/ssai-cnn/wiki/Predicted-results)

# Reference
If you use this code for your project, please cite this conference paper:
Tongchun Zuo, Juntao Feng, Xuejin Chen. "HF-FCN: Hierarchically Fused Fully Convolutional Network for Robust Building Extraction". Asian Conference of Computer Vision. 2016. 
