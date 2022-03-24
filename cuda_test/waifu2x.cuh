#pragma once

#include <cudnn.h>
//#include <cudnn_cnn_infer.h>

#include <stdexcept>

#include "picojson.h"
#include <string>
#include <fstream>

#include "debug.hpp"

#include <memory>

namespace std {
    template<>
    struct default_delete<std::remove_pointer<cudnnFilterDescriptor_t>::type> {
        void operator()(cudnnFilterDescriptor_t ptr) const {
            cudnnCheck(cudnnDestroyFilterDescriptor(ptr));
        }
    };

    template<>
    struct default_delete<std::remove_pointer<cudnnConvolutionDescriptor_t>::type> {
        void operator()(cudnnConvolutionDescriptor_t ptr) const {
            cudnnCheck(cudnnDestroyConvolutionDescriptor(ptr));
        }
    };

    template<>
    struct default_delete<std::remove_pointer<cudnnActivationDescriptor_t>::type> {
        void operator()(cudnnActivationDescriptor_t ptr) const {
            cudnnCheck(cudnnDestroyActivationDescriptor(ptr));
        }
    };

    template<>
    struct default_delete<std::remove_pointer<cudnnTensorDescriptor_t>::type> {
        void operator()(cudnnTensorDescriptor_t ptr) const {
            cudnnCheck(cudnnDestroyTensorDescriptor(ptr));
        }
    };

}


struct CNNModel {
    struct Layer {
        enum Direction {
            Unknown = 0,
            Forward,
            Backward,
        } direction_;

        std::unique_ptr<std::remove_pointer<cudnnFilterDescriptor_t>::type> filter_desc_;
        std::unique_ptr<std::remove_pointer<cudnnConvolutionDescriptor_t>::type> conv_desc_;
        std::unique_ptr<std::remove_pointer<cudnnTensorDescriptor_t>::type> bias_desc_;
        std::unique_ptr<std::remove_pointer<cudnnActivationDescriptor_t>::type> activation_desc_;

        cudnnConvolutionFwdAlgoPerf_t forward_algo_;
        cudnnConvolutionBwdDataAlgoPerf_t backward_algo_;

        std::vector<float> host_weight_;
        std::vector<float> host_bias_;

        device_unique_ptr device_weight_ptr_;
        device_unique_ptr device_bias_ptr_;

        void diff(int& h, int& w) {
            if (direction_ == Direction::Forward) {
                mypl;
                h -= 2, w -= 2;
            }
            else if (direction_ == Direction::Backward){
                mypl;
                h = 2 * h - 4, w = 2 * w - 4;
            }
        }

        Layer(Layer&& rhs) = default;
        Layer(Layer const& rhs) = delete;
        Layer& operator=(Layer const& rhs) = delete;

        int dH_, dW_, kH_, kW_, padH_, padW_, nInputPlane_, nOutputPlane_;
        Layer(Direction dir, int dH, int dW, int kH, int kW, int padH, int padW, int nInputPlane, int nOutputPlane) :
            direction_(dir)
        {
            dH_ = dH;
            dW_ = dW;
            kH_ = kH;
            kW_ = kW;
            padH_ = padH;
            padW_ = padW;
            nInputPlane_ = nInputPlane;
            nOutputPlane_ = nOutputPlane;

            host_weight_.resize(kW * kH * nInputPlane * nOutputPlane);
            host_bias_.resize(nOutputPlane);

            cudnnFilterDescriptor_t temp_filter_desc;
            cudnnCheck(cudnnCreateFilterDescriptor(&temp_filter_desc));
            filter_desc_.reset(temp_filter_desc);

            if(direction_ == Direction::Forward)
                cudnnCheck(cudnnSetFilter4dDescriptor(filter_desc_.get(), CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, nOutputPlane_, nInputPlane_, kH_, kW_));
            else if(direction_ == Direction::Backward)
                cudnnCheck(cudnnSetFilter4dDescriptor(filter_desc_.get(), CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, nInputPlane_, nOutputPlane_, kH_, kW_));


            cudnnConvolutionDescriptor_t temp_conv_desc;
            cudnnCheck(cudnnCreateConvolutionDescriptor(&temp_conv_desc));
            conv_desc_.reset(temp_conv_desc);

            cudnnCheck(cudnnSetConvolution2dDescriptor(conv_desc_.get(), padH, padW, dH, dW, 1, 1,
                cudnnConvolutionMode_t::CUDNN_CONVOLUTION, cudnnDataType_t::CUDNN_DATA_FLOAT));

            cudnnSetConvolutionMathType(conv_desc_.get(), cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);


            cudnnTensorDescriptor_t temp_bias_desc;
            cudnnCheck(cudnnCreateTensorDescriptor(&temp_bias_desc));
            bias_desc_.reset(temp_bias_desc);

            cudnnCheck(cudnnSetTensor4dDescriptor(bias_desc_.get(), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, nOutputPlane, 1, 1));

            cudnnActivationDescriptor_t temp_activation_desc;
            cudnnCheck(cudnnCreateActivationDescriptor(&temp_activation_desc));
            activation_desc_.reset(temp_activation_desc);

            cudnnCheck(cudnnSetActivationDescriptor(activation_desc_.get(), cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY,
                cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN, 0.0));
        }

        ~Layer() {
        }

        size_t algorithm_workspace(cudnnHandle_t handle, cudnnTensorDescriptor_t src, cudnnTensorDescriptor_t dst) {
            int nAlgos = 0;
            size_t workspace_size = 0;
            mypl;

            try {
                switch (direction_) {
                case Forward:
                    cudnnCheck(cudnnFindConvolutionForwardAlgorithm(
                        handle, src, filter_desc_.get(), conv_desc_.get(), dst, 1, &nAlgos, &forward_algo_));
                    forward_algo_.algo = cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
                    cudnnCheck(cudnnGetConvolutionForwardWorkspaceSize(
                        handle, src, filter_desc_.get(), conv_desc_.get(), dst, forward_algo_.algo, &workspace_size));
                    break;


                case Backward:
                    cudnnCheck(cudnnFindConvolutionBackwardDataAlgorithm(
                        handle, filter_desc_.get(), src, conv_desc_.get(), dst, 1, &nAlgos, &backward_algo_));
                    cudnnCheck(cudnnGetConvolutionBackwardDataWorkspaceSize(
                        handle, filter_desc_.get(), src, conv_desc_.get(), dst, backward_algo_.algo, &workspace_size));
                    break;
                }
            }
            catch (std::exception const& exc) {
                mypv(exc.what());

            }
            return workspace_size;
        }

        void execute(cudnnHandle_t const handle, 
            cudnnTensorDescriptor_t const src, void const* src_data,
            cudnnTensorDescriptor_t dst, void* dst_data,
            void* workspace, size_t workspace_size
        ){
            float const one = 1.f, zero = 0.f;

            switch (direction_) {
            case Forward:
                cudnnCheck(cudnnConvolutionBiasActivationForward(
                    handle,
                    &one, src, src_data,
                    filter_desc_.get(), device_weight_ptr_.get(),
                    conv_desc_.get(), forward_algo_.algo,
                    workspace, workspace_size,
                    &zero, dst, dst_data,
                    bias_desc_.get(), device_bias_ptr_.get(),
                    activation_desc_.get(),
                    dst, dst_data
                ));
                break;

            case Backward:
                cudnnCheck(cudnnConvolutionBackwardData(
                    handle, &one,
                    filter_desc_.get(), device_weight_ptr_.get(),
                    src, src_data,
                    conv_desc_.get(), backward_algo_.algo,
                    workspace, workspace_size,
                    &zero,
                    dst, dst_data
                ));
                break;
            }
        }

    };

    std::vector<Layer> layers;

    CNNModel(std::string const& file_name) {
        std::ifstream model_file(file_name.c_str());
        if (model_file.bad()) {
            throw std::runtime_error("Model file not found: " + file_name);
        }

        picojson::value json;
        model_file >> json;

        auto& model_json = json.get<picojson::array>();
        for (auto& layer_json0 : model_json) {
            auto& layer_json = layer_json0.get<picojson::object>();

            std::string const class_name = layer_json["class_name"].get<std::string>();
            CNNModel::Layer::Direction dir = CNNModel::Layer::Direction::Unknown;
            if (class_name == "nn.SpatialConvolutionMM")
                dir = CNNModel::Layer::Direction::Forward;
            else if (class_name == "nn.SpatialFullConvolution")
                dir = CNNModel::Layer::Direction::Backward;

            layers.emplace_back(
                dir,
                layer_json["dH"].get<double>() + 0.5,
                layer_json["dW"].get<double>() + 0.5,
                layer_json["kH"].get<double>() + 0.5,
                layer_json["kW"].get<double>() + 0.5,
                layer_json["padH"].get<double>() + 0.5,
                layer_json["padW"].get<double>() + 0.5,
                layer_json["nInputPlane"].get<double>() + 0.5,
                layer_json["nOutputPlane"].get<double>() + 0.5
            );
            auto& layer = layers.back();

            auto& kernels = layer_json["weight"].get<picojson::array>();

            switch (layer.direction_) {
            case CNNModel::Layer::Direction::Forward:
                for (int i = 0; i < layer.nOutputPlane_; i++) {
                    auto& kernel = kernels[i].get<picojson::array>();
                    for (int j = 0; j < layer.nInputPlane_; j++) {
                        auto& mat = kernel[j].get<picojson::array>();
                        for (int k = 0; k < layer.kH_; k++) {
                            auto& row = mat[k].get<picojson::array>();
                            for (int l = 0; l < layer.kW_; l++) {
                                layer.host_weight_[
                                    i * (layer.nInputPlane_ * layer.kH_ * layer.kW_)
                                        + j * (layer.kH_ * layer.kW_)
                                        + k * layer.kW_
                                        + l
                                ] = row[l].get<double>();
                            }
                        }
                    }
                }
                break;

            case CNNModel::Layer::Direction::Backward:
                for (int i = 0; i < layer.nInputPlane_; i++) {
                    auto& kernel = kernels[i].get<picojson::array>();
                    for (int j = 0; j < layer.nOutputPlane_; j++) {
                        auto& mat = kernel[j].get<picojson::array>();
                        for (int k = 0; k < layer.kH_; k++) {
                            auto& row = mat[k].get<picojson::array>();
                            for (int l = 0; l < layer.kW_; l++) {
                                layer.host_weight_[
                                    i * (layer.nOutputPlane_ * layer.kH_ * layer.kW_)
                                        + j * (layer.kH_ * layer.kW_)
                                        + k * layer.kW_
                                        + l
                                ] = row[l].get<double>();
                            }
                        }
                    }
                }
                break;
            }

            auto& bias = layer_json["bias"].get<picojson::array>();

            for (int i = 0; i < layer.nOutputPlane_; i++)
                layer.host_bias_[i] = bias[i].get<double>();

            layer.device_weight_ptr_ = cuda_memory_allocate(sizeof(float) * layer.host_weight_.size());
            cudaCheck(cudaMemcpy(layer.device_weight_ptr_.get(), layer.host_weight_.data(), 
                sizeof(float) * layer.host_weight_.size(), cudaMemcpyKind::cudaMemcpyHostToDevice));

            layer.device_bias_ptr_ = cuda_memory_allocate(sizeof(float) * layer.host_bias_.size());
            cudaCheck(cudaMemcpy(layer.device_bias_ptr_.get(), layer.host_bias_.data(), 
                sizeof(float) * layer.host_bias_.size(), cudaMemcpyKind::cudaMemcpyHostToDevice));
        }

        model_file.close();
    }
};
