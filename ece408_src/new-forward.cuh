#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#include <stdio.h>

#define min(a, b)   ((a < b) ? a : b)

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define w4d(i3, i2, i1, i0) w[(i3) * (C * KERNEL_WIDTH * KERNEL_WIDTH) + (i2) * (KERNEL_WIDTH * KERNEL_WIDTH) + (i1) * (KERNEL_WIDTH) + i0]

#define KERNEL_WIDTH    5
#define BLOCK_WIDTH     8
#define CHANNEL_WIDTH   16

/* layer 1 constants */
#define L1_Hin          48
#define L1_Win          48
#define L1_Cin          1
#define L1_Hout         44
#define L1_Wout         44
#define L1_Cout         6

#define L1_TILE_WIDTH   8
#define L1_BLK_WIDTH    (L1_TILE_WIDTH + KERNEL_WIDTH - 1)
#define L1_TILE_SIZE    (L1_TILE_WIDTH*L1_TILE_WIDTH)
#define L1_BLK_SIZE     (L1_BLK_WIDTH*L1_BLK_WIDTH)
#define L1_LOAD_CYCLE   ceil((float)L1_Cin * L1_BLK_SIZE / L1_TILE_SIZE)

/* layer 2 constants */
#define L2_Hin          22
#define L2_Win          22
#define L2_Cin          6
#define L2_Hout         18
#define L2_Wout         18
#define L2_Cout         16

#define L2_TILE_WIDTH   8
#define L2_BLK_WIDTH    (L2_TILE_WIDTH + KERNEL_WIDTH - 1)
#define L2_TILE_SIZE    (L2_TILE_WIDTH*L2_TILE_WIDTH)
#define L2_BLK_SIZE     (L2_BLK_WIDTH*L2_BLK_WIDTH)
#define L2_LOAD_CYCLE   ceil((float)L2_Cin * L2_BLK_SIZE / L2_TILE_SIZE)

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

__constant__ float kernel1[L1_Cout][L1_Cin][KERNEL_WIDTH][KERNEL_WIDTH];
__constant__ float kernel2[L2_Cout][L2_Cin][KERNEL_WIDTH][KERNEL_WIDTH];

__global__ void forward_layer1(float* __restrict__ y, const float* __restrict__ x, const int B) {

    __shared__ float cache[L1_Cin][L1_BLK_WIDTH][L1_BLK_WIDTH];

    static const int H        = L1_Hin;
    static const int W        = L1_Win;
    static const int C        = L1_Cin;
    static const int H_out    = L1_Hout;
    static const int W_out    = L1_Wout;
    static const int M        = L1_Cout;

    const int col   = bx * L1_TILE_WIDTH + tx;
    const int row   = by * L1_TILE_WIDTH + ty;
    const int b_col = bx * L1_TILE_WIDTH;       // base column
    const int b_row = by * L1_TILE_WIDTH;       // base row
    const int l_idx = tz * L1_TILE_SIZE + ty * L1_TILE_WIDTH + tx; // linearized thread index
    const int cout  = tz;
    const int batch = bz;

    float sum = 0;
    float tmp = 0;

    if (batch < B) {
        // load shared cache
        if (tz < L1_LOAD_CYCLE){ // use one of the threads to load
            int off_col = l_idx % L1_BLK_WIDTH;
            int off_row = l_idx / L1_BLK_WIDTH % L1_BLK_WIDTH;
            int channel = l_idx / L1_BLK_SIZE;
            if (channel < C) {
                if (b_col + off_col < W && b_row + off_row < H)
                    tmp = x4d(batch, channel, off_row+b_row, off_col+b_col);
                cache[channel][off_row][off_col] = tmp;
            }
        }

        __syncthreads();

        // perform computation
        if (row < H_out && col < W_out) {
            y4d(batch, cout, row, col) = 0;
            #pragma unroll
            for (int c = 0; c < C; ++c)
                #pragma unroll
                for (int p = 0; p < KERNEL_WIDTH; ++p)
                    #pragma unroll
                    for (int q = 0; q < KERNEL_WIDTH; ++q)
                        sum += cache[c][ty+p][tx+q] * kernel1[cout][c][p][q];
            y4d(batch, cout, row, col) = sum;
        }
    }
}

__global__ void forward_layer2(float* __restrict__ y, const float* __restrict__ x, const int B) {

    __shared__ float cache[L2_Cin][L2_BLK_WIDTH][L2_BLK_WIDTH];

    static int H        = L2_Hin;
    static int W        = L2_Win;
    static int C        = L2_Cin;
    static int H_out    = L2_Hout;
    static int W_out    = L2_Wout;
    static int M        = L2_Cout;

    const int col   = bx * L2_TILE_WIDTH + tx;
    const int row   = by * L2_TILE_WIDTH + ty;
    const int b_col = bx * L2_TILE_WIDTH;       // base column
    const int b_row = by * L2_TILE_WIDTH;       // base row
    const int l_idx = tz * L2_TILE_SIZE + ty * L2_TILE_WIDTH + tx; // linearized thread index
    const int cout  = tz;
    const int batch = bz;

    float sum = 0;
    float tmp = 0;

    if (batch < B) {
        // load shared cache
        if (tz < L2_LOAD_CYCLE){ // use one of the threads to load
            int off_col = l_idx % L2_BLK_WIDTH;
            int off_row = l_idx / L2_BLK_WIDTH % L2_BLK_WIDTH;
            int channel = l_idx / L2_BLK_SIZE;
            if (channel < C) {
                if (b_col + off_col < W && b_row + off_row < H)
                    tmp = x4d(batch, channel, off_row+b_row, off_col+b_col);
                cache[channel][off_row][off_col] = tmp;
            }
        }

        __syncthreads();

        // perform computation
        if (row < H_out && col < W_out) {
            #pragma unroll
            for (int c = 0; c < C; ++c)
                #pragma unroll
                for (int p = 0; p < KERNEL_WIDTH; ++p)
                    #pragma unroll
                    for (int q = 0; q < KERNEL_WIDTH; ++q)
                        sum += cache[c][ty+p][tx+q] * kernel2[cout][c][p][q];
            y4d(batch, cout, row, col) = sum;
        }
    }
}

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

    // create pointer aliases
    float *xptr = x.dptr_;
    float *yptr = y.dptr_;
    float *wptr = w.dptr_;

    // kernel execution
    if (C_in == L1_Cin) {
        dim3 gridDim(ceil((float)W_out / L1_TILE_WIDTH),
                     ceil((float)H_out / L1_TILE_WIDTH), B);
        dim3 blockDim(L1_TILE_WIDTH, L1_TILE_WIDTH, L1_Cout);
        cudaMemcpyToSymbol(kernel1, wptr, sizeof(float) * K_SIZE);
        forward_layer1<<<gridDim, blockDim>>>(yptr, xptr, B);
    }
    else if (C_in == L2_Cin) {
        dim3 gridDim(ceil((float)W_out / L2_TILE_WIDTH),
                     ceil((float)H_out / L2_TILE_WIDTH), B);
        dim3 blockDim(L2_TILE_WIDTH, L2_TILE_WIDTH, L2_Cout);
        cudaMemcpyToSymbol(kernel2, wptr, sizeof(float) * K_SIZE);
        forward_layer2<<<gridDim, blockDim>>>(yptr, xptr, B);
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
