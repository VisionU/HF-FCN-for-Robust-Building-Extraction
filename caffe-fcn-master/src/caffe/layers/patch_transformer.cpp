#include <opencv2/opencv.hpp>
#include "caffe/vision_layers.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void PatchTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*> &bottom,
    const vector<Blob<Dtype>*> &top) {
  for (int blob_id = 0; blob_id < bottom.size(); ++blob_id) {
    LOG(INFO) << "transformer input: "
              << bottom[blob_id]->num() << ", "
              << bottom[blob_id]->channels() << ", "
              << bottom[blob_id]->height() << ", "
              << bottom[blob_id]->width();
  }
}

template <typename Dtype>
void PatchTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> &bottom,
    const vector<Blob<Dtype>*> &top) {
  const google::protobuf::RepeatedField<uint32_t> crop_sizes =
    this->layer_param_.patch_transformer_param().crop_size();
  CHECK_EQ(crop_sizes.size(), bottom.size());
  for (int blob_id = 0; blob_id < bottom.size(); ++blob_id) {
    top[blob_id]->Reshape(bottom[blob_id]->num(), bottom[blob_id]->channels(),
                          crop_sizes.Get(blob_id), crop_sizes.Get(blob_id));
  }
}

template <typename Dtype>
void PatchTransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*> &bottom,
    const vector<Blob<Dtype>*> &top) {
  CHECK_EQ(bottom.size(), top.size());
  // crop sizes
  const google::protobuf::RepeatedField<uint32_t> crop_sizes =
    this->layer_param_.patch_transformer_param().crop_size();
  CHECK_EQ(crop_sizes.size(), bottom.size());
  // binarize
  const google::protobuf::RepeatedField<bool> binarize =
    this->layer_param_.patch_transformer_param().binarize();
  if (binarize.size() > 0)
    CHECK_EQ(binarize.size(), bottom.size());

  for (int i = 0; i < bottom[0]->num(); ++i) {
    vector<cv::Mat> imgs;
    // both rotation and cropping are performed to all bottom blobs
    const float angle = static_cast<float>(caffe_rng_rand() % 360);
    // flip_code takes the value ranging 0, 1
    // const int flip_code = caffe_rng_rand() % 2;
    
    const int flip_code = caffe_rng_rand() % 3; //caffe-fcn
      
    for (int blob_id = 0; blob_id < bottom.size(); ++blob_id) {
      const int channels = bottom[blob_id]->channels();
      const int height = bottom[blob_id]->height();
      const int width = bottom[blob_id]->width();
      const Dtype *data = bottom[blob_id]->cpu_data()
                          + bottom[blob_id]->offset(i);
      cv::Mat img = ConvertToCVMat(data, channels, height, width);
        
        
      // randomly flipping (when flip_code == 0, it's disabled)
      if (this->layer_param_.patch_transformer_param().flip() //caffe-fcn
          && (flip_code == 1 || flip_code == 2 )) {
        cv::flip(img, img, (flip_code % 2));
      }

      // randomly rotate
      if (this->layer_param_.patch_transformer_param().rotate()) {
        cv::Point2f pt(width / 2.0, height / 2.0);
        cv::Mat rot = cv::getRotationMatrix2D(pt, angle, 1.0);
        cv::warpAffine(img, img, rot, cv::Size(width, height),
                       cv::INTER_NEAREST);
      }

      // crop center
      const int size = crop_sizes.Get(blob_id);
      cv::Mat patch(size, size, CV_32FC(channels));
      img(cv::Rect(width / 2 - size / 2,
                   height / 2 - size / 2, size, size)).copyTo(patch);

      // binarization
      if (binarize.Get(blob_id)) {
        cv::Mat *slice = new cv::Mat[bottom[blob_id]->channels()];
        cv::split(patch, slice);
        for (int c = 0; c < bottom[blob_id]->channels(); ++c) {
          cv::Mat tmp = slice[c].clone();
          cv::threshold(tmp, slice[c], 0.5, 1, cv::THRESH_BINARY);
        }
        cv::merge(slice, bottom[blob_id]->channels(), patch);
        delete [] slice;
      }
      imgs.push_back(patch);
    }

    // patch-wise mean subtraction
    cv::Scalar mean, stddev;
    cv::meanStdDev(imgs[0], mean, stddev);
    if (this->layer_param_.patch_transformer_param().mean_normalize()) {
      cv::Mat *slice = new cv::Mat[bottom[0]->channels()];
      cv::split(imgs[0], slice);
      for (int c = 0; c < bottom[0]->channels(); ++c) {
        cv::subtract(slice[c], mean[c], slice[c]);
      }
      cv::merge(slice, bottom[0]->channels(), imgs[0]);
      delete [] slice;
    }

    // patch-wise stddev division
    if (this->layer_param_.patch_transformer_param().stddev_normalize()) {
      cv::Mat *slice = new cv::Mat[bottom[0]->channels()];
      cv::split(imgs[0], slice);
      for (int c = 0; c < bottom[0]->channels(); ++c) {
        slice[c] /= stddev[c];
      }
      cv::merge(slice, bottom[0]->channels(), imgs[0]);
      delete [] slice;
    }

    // constant value subtraction
    const google::protobuf::RepeatedField<float> subs =
      this->layer_param_.patch_transformer_param().subtract();
    if (subs.size() > 0) {
      CHECK_EQ(subs.size(), bottom[0]->channels());
      vector<cv::Mat> splitted;
      cv::split(imgs[0], splitted);
      for (int j = 0; j < splitted.size(); ++j) {
        splitted.at(j).convertTo(splitted.at(j), CV_32F, 1.0, -subs.Get(j));
      }
      cv::merge(splitted, imgs[0]);
    }

    // stddev division
    const google::protobuf::RepeatedField<float> divs =
      this->layer_param_.patch_transformer_param().divide();
    if (divs.size() > 0) {
      CHECK_EQ(divs.size(), bottom[0]->channels());
      vector<cv::Mat> splitted;
      cv::split(imgs[0], splitted);
      for (int j = 0; j < splitted.size(); ++j) {
        splitted.at(j).convertTo(splitted.at(j), CV_32F, 1.0 / divs.Get(j));
      }
      cv::merge(splitted, imgs[0]);
    }

    // revert into blob
    for (int blob_id = 0; blob_id < bottom.size(); ++blob_id) {
      ConvertFromCVMat(imgs[blob_id], top[blob_id]->mutable_cpu_data()
                       + top[blob_id]->offset(i));
    }
  }
}

template <typename Dtype>
cv::Mat PatchTransformerLayer<Dtype>::ConvertToCVMat(
  const Dtype *data, const int &channels,
  const int &height, const int &width) {
  cv::Mat img(height, width, CV_32FC(channels));
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        int index = c * height * width + h * width + w;
        float val = static_cast<float>(data[index]);
        int pos = h * width * channels + w * channels + c;
        reinterpret_cast<float *>(img.data)[pos] = val;
      }
    }
  }

  return img;
}

template <typename Dtype>
void PatchTransformerLayer<Dtype>::ConvertFromCVMat(const cv::Mat img, Dtype *data) {
  const int channels = img.channels();
  const int height = img.rows;
  const int width = img.cols;
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        const int pos = h * width * channels + w * channels + c;
        float val = reinterpret_cast<float *>(img.data)[pos];
        const int index = c * height * width + h * width + w;
        data[index] = static_cast<Dtype>(val);
      }
    }
  }
}

INSTANTIATE_CLASS(PatchTransformerLayer);
REGISTER_LAYER_CLASS(PatchTransformer);

}  // namespace caffe