#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#include <stdio.h>

#define min(a, b)   ((a < b) ? a : b)

#define KERNEL_WIDTH    5
#define KERNEL_M        16
#define KERNEL_C        6
#define BLOCK_WIDTH     8
#define CHANNLE_SIZE    16

#define BATCH_SIZE      256

#define bx  blockIdx.x
#define by  blockIdx.y
#define bz  blockIdx.z
#define tx  threadIdx.x
#define ty  threadIdx.y
#define tz  threadIdx.z

namespace mxnet
{
namespace op
{

__constant__ float kernel[KERNEL_M*KERNEL_C*KERNEL_WIDTH*KERNEL_WIDTH];

__global__ void forward_kernel(float *y, const float *x,
        const int B, const int M, const int C, const int H, const int W, const int K) {

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int col     = bx * BLOCK_WIDTH + tx;
    int row     = by * BLOCK_WIDTH + ty;
    int zdim    = bz * CHANNLE_SIZE + tz;
    int channel = zdim % M;
    int batch   = zdim / M;

    if (batch < B && row < H_out && col < W_out) {
        float sum = 0;
        for (int c = 0; c < C; c++)
            for (int p = 0; p < K; ++p)
                for (int q = 0; q < K; ++q)
                    sum += x4d(batch, c, row + p, col + q) * k4d(channel, c, p, q);
        y4d(batch, channel, row, col) = sum;
    }

#undef y4d
#undef x4d
#undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y,
        const mshadow::Tensor<gpu, 4, float> &x,
        const mshadow::Tensor<gpu, 4, float> &w) {

    // Use mxnet's CHECK_EQ to do assertions.

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int K         = KERNEL_WIDTH;
    const int B         = x.shape_[0];
    const int C_in      = x.shape_[1];
    const int C_out     = y.shape_[1];
    const int H_in      = x.shape_[2];
    const int W_in      = x.shape_[3];
    const int H_out     = y.shape_[2];
    const int W_out     = y.shape_[3];
    const int K_SIZE    = K * K * C_in * C_out;

    // Create CUDA streams
    const int i_size = H_in * W_in * C_in;
    const int o_size = H_out * W_out * C_out;
    const int num_stream = ceil((float)B / BATCH_SIZE);
    cudaStream_t stream[num_stream];

    // Set the kernel dimensions
    dim3 gridDim(ceil((float)W_out / BLOCK_WIDTH),
                 ceil((float)H_out / BLOCK_WIDTH),
                 ceil((float)C_out * BATCH_SIZE / CHANNLE_SIZE));
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, CHANNLE_SIZE);

    // Call the kernel
    float *xptr = x.dptr_;
    float *yptr = y.dptr_;
    float *wptr = w.dptr_;
    cudaMemcpyToSymbol(kernel, wptr, sizeof(float) * K_SIZE);
    for (int i = 0; i < num_stream; ++i) {
        cudaStreamCreate(&stream[i]);
        forward_kernel<<<gridDim, blockDim, 0, stream[i]>>>(
                yptr + i * BATCH_SIZE * o_size,
                xptr + i * BATCH_SIZE * i_size,
                min(BATCH_SIZE, B - i * BATCH_SIZE), C_out, C_in, H_in, W_in, K);
    }

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
