# This repo has been deprecated because whole things are re-implemented by using Chainer and I did refactoring for many codes. So please check this newer version: https://github.com/mitmul/ssai-cnn

# Semantic Segmentation for Aerial Imagery
Extract building and road from aerial imagery

# Requirements
- OpenCV 2.4.10
- Boost 1.57.0
- Boost.NumPy
- Caffe (modified caffe: [https://github.com/mitmul/caffe](https://github.com/mitmul/caffe))
  - NOTE: Build the `ssai` branch of the above repository

# Data preparation

```
$ bash shells/donwload.sh
$ python scripts/create_dataset.py --dataset multi
$ python scripts/create_dataset.py --dataset single
$ python scripts/create_dataset.py --dataset roads_mini
$ python scripts/create_dataset.py --dataset roads
$ python scripts/create_dataset.py --dataset buildings
$ python scripts/create_dataset.py --dataset merged
```

## Massatusetts Building & Road dataset
- mass_roads
  - train: 8458173 patches
    - epoch: 66079 mini-batches (mini-batch size: 128)

  - valid: 126281 patches
    - epoch: 987 mini-batches (mini-batch size: 128)

  - test: 440932 patches
    - epoch: 3445 mini-batches (mini-batch size: 128)

- mass_roads_mini, mass_buildings, mass_merged
  - train: 1119872 patches
    - epoch: 8749 mini-batches (mini-batch size: 128)

  - valid: 36100 patches
    - epoch: 282 mini-batches (mini-batch size: 128)

  - test: 89968 patches
    - epoch: 703 mini-batches (mini-batch size: 128)

# Create Models

```
$ python scripts/create_models.py --seed seeds/model_seeds.json --caffe_dir $HOME/lib/caffe/build/install
```

# Start training

```
$ bash shells/train.sh models/Mnih_CNN
```

will create a directory named `results/Mnih_CNN_{started date}`.

# Prediction

```
$ cd results/Mnih_CNN_{started date}
$ python ../../scripts/test_prediction.py --model predict.prototxt --weight snapshots/Mnih_CNN_iter_1000000.caffemodel --img_dir ../../data/mass_merged/test/sat --channel 3
```

# Build Library for Evaluation

```
$ cd lib
$ mkdir build
$ cd build
$ cmake ../
$ make
```

# Evaluation

```
$ cd results/Mnih_CNN_{started date}
$ python ../../scripts/test_evaluation.py --map_dir ../../data/mass_merged/test/map --result_dir prediction_1000000 --channel 3
```

# Model averaging

```
$ python ../scripts/batch_evaluation.py --offset True
$ mkdir Mnih_CNN_Merged
$ cd Mnih_CNN_Merged
$ python ../../scripts/test_evaluation.py --map_dir ../../data/mass_merged/test/map --result_dir ./prediction_100000 --channel 3 --offset 0 --pad 31
```
