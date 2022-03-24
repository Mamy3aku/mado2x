#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>

#include "cuda_runtime.h"

#define mypv(val) do{auto const& xx = val; std::cerr << #val << '=' << (xx) << std::endl;}while(false)
#define mypl do{std::cerr << '@' << __LINE__ << std::endl;}while(false)

#define cudnnCheck(x) do{auto const status = x; if(status){ throw std::runtime_error(std::string(cudnnGetErrorString(status)) + ':' + __FILE__ + '@' + std::to_string(__LINE__)); } }while(false)
#define cudaCheck(x) do{auto const status = x; if(status){ throw std::runtime_error(std::string(cudaGetErrorName(status)) + ':' + __FILE__ + '@' + std::to_string(__LINE__)); } }while(false)

struct cuda_device_memory_delete {
    void operator()(void* ptr) const {
        mypl;
        cudaCheck(cudaFree(ptr));
    }
};
using device_unique_ptr = std::unique_ptr<void, cuda_device_memory_delete>;

device_unique_ptr cuda_memory_allocate(size_t n) {
    void* ptr = nullptr;
    cudaCheck(cudaMalloc(&ptr, n));

    return device_unique_ptr(ptr);
}


