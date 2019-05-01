#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#include <stdio.h>

#define min(a, b)   ((a < b) ? a : b)

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

#define KERNEL_WIDTH    5

/* layer 1 constants -- matrix unrolled constants */
#define L1_Hin          48
#define L1_Win          48
#define L1_Cin          1
#define L1_Hout         44
#define L1_Wout         44
#define L1_Cout         6

#define L1_TILE_HEIGHT  11
#define L1_TILE_WIDTH   44
#define L1_BLK_HEIGHT   (L1_TILE_HEIGHT + KERNEL_WIDTH - 1)
#define L1_BLK_WIDTH    (L1_TILE_WIDTH + KERNEL_WIDTH - 1)
#define L1_TILE_SIZE    (L1_TILE_HEIGHT*L1_TILE_WIDTH)
#define L1_BLK_SIZE     (L1_BLK_HEIGHT*L1_BLK_WIDTH)
/* not using ceiling to ensure contant evaluation by preprocessor */
#define L1_LOAD_CYCLE   (L1_Cin * L1_BLK_SIZE / L1_TILE_SIZE + 1)

/* layer 2 constants */
#define L2_Hin          22
#define L2_Win          22
#define L2_Cin          6
#define L2_Hout         18
#define L2_Wout         18
#define L2_Cout         16

#define L2_TILE_HEIGHT  18
#define L2_TILE_WIDTH   18
#define L2_BLK_HEIGHT   (L2_TILE_HEIGHT + KERNEL_WIDTH - 1)
#define L2_BLK_WIDTH    (L2_TILE_WIDTH + KERNEL_WIDTH - 1)
#define L2_TILE_SIZE    (L2_TILE_HEIGHT*L2_TILE_WIDTH)
#define L2_BLK_SIZE     (L2_BLK_HEIGHT*L2_BLK_WIDTH)
/* not using ceiling to ensure contant evaluation by preprocessor */
#define L2_LOAD_CYCLE   (L2_Cin / 2 * L2_BLK_SIZE / L2_TILE_SIZE + 1)

#define bx  blockIdx.x
#define by  blockIdx.y
#define bz  blockIdx.z
#define tx  threadIdx.x
#define ty  threadIdx.y
#define tz  threadIdx.z

__constant__ float kernel1[L1_Cout][L1_Cin][KERNEL_WIDTH][KERNEL_WIDTH];
//__constant__ half2 kernel1[L1_Cout/2][L1_Cin][KERNEL_WIDTH][KERNEL_WIDTH];
__constant__ half2 kernel2[L2_Cout/2][L2_Cin][KERNEL_WIDTH][KERNEL_WIDTH];

namespace mxnet
{
namespace op
{

__global__ void forward_layer1(const float* __restrict__ y, float* __restrict__ x, const int B) {
    __shared__ float cache[L1_TILE_HEIGHT][L1_TILE_WIDTH]
}

__global__ void forward_layer2(float* __restrict__ y, const float* __restrict__ x, const int B) {

    __shared__ half2 cache[L2_Cin/2][L2_BLK_HEIGHT][L2_BLK_WIDTH];

    static int H        = L2_Hin;
    static int W        = L2_Win;
    static int C        = L2_Cin;
    static int H_out    = L2_Hout;
    static int W_out    = L2_Wout;
    static int M        = L2_Cout;

    const int col   = bx * L2_TILE_WIDTH + tx;
    const int row   = by * L2_TILE_HEIGHT + ty;
    const int b_col = bx * L2_TILE_WIDTH;       // base column
    const int b_row = by * L2_TILE_HEIGHT;       // base row
    const int l_idx = ty * L2_TILE_WIDTH + tx; // linearized thread index
    const int batch = bz;

    half2 value;
    half2 sum[L2_Cout/2] = {{ 0, 0 }};
    half2 tmp;

    int off_idx, off_col, off_row, channel;

    if (batch < B) {
        // load shared cache
        #pragma unroll
        for (int i = 0; i < L2_LOAD_CYCLE; ++i) {
            value.x = value.y = 0;
            off_idx = l_idx + i * L2_TILE_SIZE;
            off_col = off_idx % L2_BLK_WIDTH;
            off_row = off_idx / L2_BLK_WIDTH % L2_BLK_HEIGHT;
            channel = off_idx / L2_BLK_SIZE;
            if (channel < C/2) {
                if (b_col + off_col < W && b_row + off_row < H) {
                    value.x = x4d(batch, channel, off_row+b_row, off_col+b_col);
                    value.y = x4d(batch, channel+C/2, off_row+b_row, off_col+b_col);
                }
                cache[channel][off_row][off_col] = value;
            }
        }

        __syncthreads();

        // perform computation
        if (row < H_out && col < W_out) {
            #pragma unroll
            for (int c = 0; c < L2_Cin/2; ++c) {
                #pragma unroll
                for (int p = 0; p < KERNEL_WIDTH; ++p) {
                    #pragma unroll
                    for (int q = 0; q < KERNEL_WIDTH; ++q) {
                        tmp.x = tmp.y = cache[c][ty+p][tx+q].x;
                        #pragma unroll
                        for (int cout = 0; cout < L2_Cout/2; ++cout) {
                            sum[cout] += tmp * kernel2[cout][c][p][q];
                        }
                    }
                }
                #pragma unroll
                for (int p = 0; p < KERNEL_WIDTH; ++p) {
                    #pragma unroll
                    for (int q = 0; q < KERNEL_WIDTH; ++q) {
                        tmp.x = tmp.y = cache[c][ty+p][tx+q].y;
                        #pragma unroll
                        for (int cout = 0; cout < L2_Cout/2; ++cout) {
                            sum[cout] += tmp * kernel2[cout][c+L2_Cin/2][p][q];
                        }
                    }
                }
            }
            #pragma unroll
            for (int cout = 0; cout < L2_Cout/2; ++cout) {
                y4d(batch, cout, row, col) = sum[cout].x;
                y4d(batch, cout+L2_Cout/2, row, col) = sum[cout].y;
            }
        }
    }
}

__global__ void float_2_half(const float* __restrict__ input, half* __restrict__ output, int size) {
    int idx_i = blockDim.x * bx + tx;
    int idx_o = 2 * idx_i;
    int half_size = size / 2;

    if (idx_i < half_size) {
        output[idx_o] = input[idx_i];
        output[idx_o+1] = input[idx_i + half_size];
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
        /* actual computation */
        dim3 gridDim(ceil((float)W_out / L1_TILE_WIDTH),
                     ceil((float)H_out / L1_TILE_HEIGHT), B);
        dim3 blockDim(L1_TILE_WIDTH, L1_TILE_HEIGHT, 1);
        cudaMemcpyToSymbol(kernel1, wptr, sizeof(float) * K_SIZE);
        forward_layer1<<<gridDim, blockDim>>>(yptr, xptr, B);
    }
    else if (C_in == L2_Cin) {
        /* float 2 half conversion */
        half *wptr_half;
        cudaMalloc(&wptr_half, sizeof(half) * K_SIZE);
        float_2_half<<<512, ceil(K_SIZE/2/512.)>>>(wptr, wptr_half, K_SIZE);
        /* actual computation */
        dim3 gridDim(ceil((float)W_out / L2_TILE_WIDTH),
                     ceil((float)H_out / L2_TILE_HEIGHT), B);
        dim3 blockDim(L2_TILE_WIDTH, L2_TILE_HEIGHT, 1);
        cudaMemcpyToSymbol(kernel2, wptr_half, sizeof(half) * K_SIZE);
        forward_layer2<<<gridDim, blockDim>>>(yptr, xptr, B);
        cudaFree(wptr_half);
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
