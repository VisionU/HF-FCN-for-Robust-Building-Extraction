#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
template<typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype> *>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template<typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype> *>& bottom,
  const vector<Blob<Dtype> *>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  loss_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                bottom[0]->height(), bottom[0]->width());

  // Check the shapes of data and label
  CHECK_EQ(bottom[0]->num(),      bottom[1]->num())
    << "The number of num of data and label should be same.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
    << "The number of channels of data and label should be same.";
  CHECK_EQ(bottom[0]->height(),   bottom[1]->height())
    << "The heights of data and label should be same.";
  CHECK_EQ(bottom[0]->width(),    bottom[1]->width())
    << "The width of data and label should be same.";
}

template<typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype> *>& bottom,
  const vector<Blob<Dtype> *>& top) {
  softmax_bottom_vec_[0] = bottom[0];

  // input details
  const int count       = bottom[0]->count();
  const int num         = bottom[0]->num();
  const int dim         = bottom[0]->count() / num;
  const int spatial_dim = bottom[0]->width() * bottom[0]->height();
  const int channels    = bottom[0]->channels();

  // all units in this channel goes to zero (GU)
  const int zero_channel =
    this->layer_param_.softmax_cross_entropy_loss_param().zero_channel();

  if (zero_channel >= 0) {
    Dtype *data = softmax_bottom_vec_[0]->mutable_cpu_data();

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        const int index = i * dim + zero_channel * spatial_dim + j;
        data[index] = 0.0;
      }
    }
  }

  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  // Stable version of loss computation from input data
  const Dtype *data  = prob_.cpu_data();
  const Dtype *label = bottom[1]->cpu_data();
  Dtype loss         = 0;

  // Compute the loss (negative log likelihood)
  const google::protobuf::RepeatedField<float> weights =
    this->layer_param_.softmax_cross_entropy_loss_param().weights();

  // If weights.Get(0) == 0 Ignoring the Loss of no interest (IL)
  if (weights.size() > 0) {
    CHECK_EQ(weights.size(), bottom[0]->channels());

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        for (int c = 0; c < channels; ++c) {
          const int index = i * dim + c * spatial_dim + j;
          CHECK_GE(label[index], 0);
          CHECK_LE(label[index], 1);
          CHECK_GE(data[index], 0);
          CHECK_LE(data[index], 1);
          loss -= weights.Get(c) * label[index] *
                  log(std::max(data[index], Dtype(kLOG_THRESHOLD)));
        }
      }
    }
  }

  // Normal negative log likelihood as the loss
  else {
    for (int i = 0; i < count; ++i) {
      CHECK_GE(label[i], 0);
      CHECK_LE(label[i], 1);
      CHECK_GE(data[i], 0);
      CHECK_LE(data[i], 1);
      loss -= label[i] * log(std::max(data[i], Dtype(kLOG_THRESHOLD)));
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template<typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype> *>& top,
  const vector<bool>         & propagate_down,
  const vector<Blob<Dtype> *>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  if (propagate_down[0]) {
    // First, compute the diff
    const int count       = bottom[0]->count();
    const int num         = bottom[0]->num();
    const int dim         = bottom[0]->count() / num;
    const int spatial_dim = bottom[0]->width() *
                            bottom[0]->height();
    const int channels = bottom[0]->channels();
    const Dtype *data  = prob_.cpu_data();
    const Dtype *label = bottom[1]->cpu_data();
    Dtype *diff        = bottom[0]->mutable_cpu_diff();

    // Grounding Units of no interest (GU)
    const int zero_channel =
      this->layer_param_.softmax_cross_entropy_loss_param().zero_channel();

    // If weights.Get(0) == 0 Ignoring the Loss of no interest (IL)
    const google::protobuf::RepeatedField<float> weights =
      this->layer_param_.softmax_cross_entropy_loss_param().weights();

    if (weights.size() > 0) {
      CHECK_EQ(weights.size(), bottom[0]->channels());

      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < spatial_dim; ++j) {
          for (int c = 0; c < channels; ++c) {
            const int index = i * dim + c * spatial_dim + j;

            if ((zero_channel >= 0) && (c == zero_channel)) {
              diff[index] = 0;
            }
            else if (weights.Get(c) == 0) {
              Dtype c_sum = 0;

              for (size_t s = 0; s < channels; s++) {
                c_sum += weights.Get(s) * label[i * dim + s * spatial_dim + j];
              }
              diff[index] = data[index] * c_sum - weights.Get(c) * label[index];
            }
            else diff[index] = data[index] - label[index];
          }
        }
      }
    }

    if ((weights.size() == 0) && (zero_channel < 0)) {
      caffe_sub(count, data, label, diff);
    }

    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxCrossEntropyLossLayer);
#endif // ifdef CPU_ONLY

INSTANTIATE_CLASS(SoftmaxCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SoftmaxCrossEntropyLoss);
} // namespace caffe
