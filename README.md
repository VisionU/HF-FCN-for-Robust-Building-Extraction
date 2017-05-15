This is a state-of-the-art project for building extraction in high resolution remote sensing image using dataset [Massachusetts road & building dataset](https://www.cs.toronto.edu/~vmnih/data/) . And, our approach was published in ACCV 2016, clik here to download [our paper](https://link.springer.com/chapter/10.1007/978-3-319-54181-5_19)

# Requirements
- caffe-fcn-master
- OpenCV 2.4.13
- CUDA V8.0
- CUDNN V5.0
- Protobuf 3.2.0 (please use this version, too low will lead to errors, like "_has_bits is not defined in this scope")
- Boost 1.59.0
- Boost.NumPy
- ssai-lib

## caffe-fcn-master
```sh
cd cmake 
cmake ..
make -j16

```

## Boost 1.59.0
```sh
tar zxvf boost_1_59_0.tar.gz 
cd boost_1_59_0 
./bootstrap.sh --with-libraries=all --with-toolset=gcc 
./b2 toolset=gcc 
sudo ./b2 install --prefix=/usr 
sudo ldconfig
```

## Boost.NumPy
```sh
git clone https://github.com/ndarray/Boost.NumPy.git 
cd Boost.Numpy  
mkdir build 
cd build 
cmake ..   
vim ../CMakeLists.txt   

add some codes before find_package(Boost COMPONENTS Python REQUIRED)  
set(BOOST_ROOT “/usr/include/boost”) 
set(Boost_LIBRARIES “/usr/include/boost/lib”)   
set(Boost_INCLUDE_DIRS “/usr/include/boost/include”) 
set(BOOST_LIBRARYDIR “/usr/include/boost/lib”) 

sudo make 
sudo make install 
```
## ssai-lib
```sh
cd ssai-lib/
mkdir build
cd build
cmake ..
make 
```
# Create Dataset
```sh
sh shells/download_minh_dataset.sh  
python scripts/create_dataset_256.py  -d /data
python scripts/verify_dataset.py -d /data/mass_building/lmdb/train_sat_256 
```  
# Start Training
```sh
cd models/HF-FCN_Models/BasicNet/  
nohup python solve.py& 
```

# Prediction
```sh
cd results/  
python ../scripts/run_prediction.py 
		 --model ../models/HF-FCN_Models/BasicNet/predict.prototxt  
		 --weight ../weights/HF-FCN_iter_12000.caffemodel  
		 --img_dir /data/mass_buildings/source/test/sat  
```
# Evaluation
```sh
python  ../scripts/run_evaluation_complex.py  \
	--gt_dir ../results/GroundTruth   \
	--pred_dir ../results/whole_image_results/Zuo-HF-FCN-ACCV16  \
	--pad 0 \
	--relax 0 \
	--pr_dir ../results/whole_image_results/PresicionRecallComparision

```

# Results Comparision
## whole image comparision
```sh
cd shells/
sh run_pr_curve_comparision.sh
```
|                                                | Recall ( \rho = 3 ) | Recall ( \rho = 0) | Time (s) |
|------------------------------------------------|---------------------|---------------------|----------|
| Mnih-CNN \cite{Mnih2013Machine}                | 0.9271              | 0.7661              | 8.70     |
| Mnih-CNN+CRF \cite{Mnih2013Machine}            | 0.9282              | 0.7638              | 26.60    |
| Saito-multi-MA \cite{Saito2016Multiple}        | 0.9503              | 0.7873              | 67.72    |
| Saito-multi-MA&CIS \cite{Saito2016Multiple} | 0.9509              | 0.7872              | 67.84    |
| *Ours (HF-FCN)                                  | 0.9643              | 0.8424              |   1.07   |*

## challenge patches comparision
```sh
cd shells/
sh run_recall_comp_for_challenge_patches.sh
```
| Method\Image ID                                      | 01             | 02             | 03             | 04             | 05             | 06             | 07             | mean           |
|-----------------------------------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Mnih-CNN+CRF\cite\{Mnih2013Machine\}          | 0.784          | 0.869          | 0.769          | 0.653          | 0.893          | 0.764          | 0.800          | 0.784          |
| Saito-multi-MA\&CIS\cite\{Saito2016Multiple\} | 0.773          | 0.915          | 0.857          | 0.789          | 0.945          | 0.773          | 0.830          | 0.851          |
| *Ours (HF-FCN)                        | 0.874 | 0.964 | 0.899 | 0.901 | 0.986 | 0.840| 0.851 | 0.911 |*


# Pre-trained models
[HF-FCN16-iter-12000.caffemodel](https://github.com/tczuo/HF-FCN-for-Robust-Building-Extraction/tree/master/weights) <br />
Minh13-Machine.caffemodel   <br />
[Saito16-Multiple-caffemodels](https://github.com/mitmul/ssai-cnn/wiki/Pre-trained-models)

# Predicted results
[HF-FCN16-results](https://github.com/tczuo/HF-FCN-for-Robust-Building-Extraction/tree/master/results/whole_image_results/Zuo-HF-FCN-ACCV16)   <br />
[Mnih13-Machine-results](https://github.com/tczuo/HF-FCN-for-Robust-Building-Extraction/tree/master/results/whole_image_results/Mnih-Machine-PHDthesis13)   <br />
[Saito16-Multiple-results](https://github.com/mitmul/ssai-cnn/wiki/Predicted-results)

# Reference
If you use this code for your project, please cite this conference paper:  <br />
@inproceedings{zuo2016hf,
  title={HF-FCN: Hierarchically Fused Fully Convolutional Network for Robust Building Extraction},
  author={Zuo, Tongchun and Feng, Juntao and Chen, Xuejin},
  booktitle={Asian Conference on Computer Vision},
  pages={291--302},
  year={2016},
  organization={Springer}
}
