This is a state-of-the-art project for building extraction in high resolution remote sensing image using dataset [Massachusetts road & building dataset](https://www.cs.toronto.edu/~vmnih/data/) . And, our approach was published in ACCV 2016, clik here to download [our paper](https://github.com/tczuo/HF-FCN-for-Robust-Building-Extraction/blob/master/0663.pdf)

# Requirements
- Last version caffe
- OpenCV 3.0.0
- NumPy
- CUDA V8.0
- CUDNN V5.0

# Create Dataset
  $ sh shells/download_minh_dataset.sh  <br />
  $ python scripts/create_dataset_256.py <br />
  
# Start Training
  $ cd models/HF-FCN_Models/BasicNet/  <br />
  
  $ nohup python solve.py& <br />

# Prediction

# Evaluation

# Results Display

# Pre-trained models and Predicted results


# Reference
If you use this code for your project, please cite this conference paper:
Tongchun Zuo, Juntao Feng, Xuejin Chen. "HF-FCN: Hierarchically Fused Fully Convolutional Network for Robust Building Extraction". Asian Conference of Computer Vision. 2016. 
