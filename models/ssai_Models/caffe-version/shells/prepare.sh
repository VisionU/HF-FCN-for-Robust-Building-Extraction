#! /bin/bash

cd lib
if [ ! -d build ]; then
	mkdir build
fi
cd build

cmake \
-DCaffe_INCLUDE_DIR=$CAFFE_ROOT/build/install/include \
-DCaffe_LIBRARY_DIR=$CAFFE_ROOT/root/caffe/build/install/lib \
 ../

make
