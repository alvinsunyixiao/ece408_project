#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__constant__ float kernels[1250];

template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y, const DType *x) {
  /*
      Modify this function to implement the forward pass described in Chapter 16.
      We have added an additional dimension to the tensors to support an entire mini-batch
      The goal here is to be correct AND fast.
      We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
      */
    __shared__ DType shared_x[784];

    int n = blockIdx.x;
    int h = threadIdx.y;
    int w = threadIdx.x;
    DType acc = 0.0;
    shared_x[h*28 + w] = x[n*784 + h*28 + w];
    __syncthreads();

    if (h < 24 && w < 24) {
        #pragma unroll 15
        for (int m = 0; m < 50; ++m) {
            acc = 0.0;
            for (int p = 0; p < 5; ++p) {
                acc += shared_x[(h+p)*28 + (w+0)] * kernels[m*25 + p*5 + 0];
                acc += shared_x[(h+p)*28 + (w+1)] * kernels[m*25 + p*5 + 1];
                acc += shared_x[(h+p)*28 + (w+2)] * kernels[m*25 + p*5 + 2];
                acc += shared_x[(h+p)*28 + (w+3)] * kernels[m*25 + p*5 + 3];
                acc += shared_x[(h+p)*28 + (w+4)] * kernels[m*25 + p*5 + 4];
            }
            y[n*28800 + m*576 + h*24 + w] = acc;
        }
    }
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    const int B = x.shape_[0];

    dim3 blockDim(28, 28, 1);
    dim3 gridDim(B, 1, 1);

    cudaMemcpyToSymbol(kernels, w.dptr_, 1250 * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    // Call the kernel
    forward_kernel<gpu, DType><<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

}
}

#endif
