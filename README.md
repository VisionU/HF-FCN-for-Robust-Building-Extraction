This is a state-of-the-art project for building extraction in high resolution remote sensing image using dataset [Massachusetts road & building dataset](https://www.cs.toronto.edu/~vmnih/data/) . And, our approach was published in ACCV 2016, clik here to download [our paper](https://github.com/tczuo/HF-FCN-for-Robust-Building-Extraction/blob/master/0663.pdf)

# Requirements
- Last version caffe
- OpenCV 2.4.13
- CUDA V8.0
- CUDNN V5.0
- Boost 1.59.0
- Boost.NumPy

## Boost 1.59.0
$ tar zxvf boost_1_59_0.tar.gz <br />
$ cd boost_1_59_0 <br />
$ ./bootstrap.sh --with-libraries=all --with-toolset=gcc <br />
$ ./b2 toolset=gcc <br />
$ sudo ./b2 install --prefix=/usr <br />
$ sudo ldconfig

## Boost.NumPy
$ git clone https://github.com/ndarray/Boost.NumPy.git <br />
$ cd Boost.Numpy  <br />
$ mkdir build <br />
$ cd build <br />
$ cmake ..   <br />
$ vim ../CMakeLists.txt   <br />
add some codes before find_package(Boost COMPONENTS Python REQUIRED)  <br />
set(BOOST_ROOT “/usr/include/boost”)  <br />
set(Boost_LIBRARIES “/usr/include/boost/lib”)   <br />
set(Boost_INCLUDE_DIRS “/usr/include/boost/include”)  <br />
set(BOOST_LIBRARYDIR “/usr/include/boost/lib”)  <br />
$ make <br />
$ make install 

# Create Dataset
$ sh shells/download_minh_dataset.sh  <br />
$ python scripts/create_dataset_256.py  <br />
$ python scripts/verify_dataset.py -d /data/mass_building/lmdb/train_sat_256  <br />
  
# Start Training
$ cd models/HF-FCN_Models/BasicNet/  <br />
$ nohup python solve.py&  <br />

# Prediction
$ cd results/   <br />
> $ python ../scripts/run_prediction.py   <br />
           --model ../models/HF-FCN_Models/BasicNet/predict.prototxt   <br />
 	         --weight ../weights/HF-FCN_iter_12000.caffemodel   <br />
      	   --img_dir /data/mass_buildings/source/test/sat   <br />

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
HF-FCN16-iter-12000.caffemodel   <br />
Minh13-Machine.caffemodel   <br />
[Saito16-Multiple-caffemodels](https://github.com/mitmul/ssai-cnn/wiki/Pre-trained-models)

# Predicted results
HF-FCN16-results   <br />
Mnih13-Machine-results   <br />
[Saito16-Multiple-results](https://github.com/mitmul/ssai-cnn/wiki/Predicted-results)

# Reference
If you use this code for your project, please cite this conference paper:
Tongchun Zuo, Juntao Feng, Xuejin Chen. "HF-FCN: Hierarchically Fused Fully Convolutional Network for Robust Building Extraction". Asian Conference of Computer Vision. 2016. 
