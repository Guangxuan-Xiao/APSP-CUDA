// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
const int BLOCK_SIZE = 100000;
namespace
{
    __global__ void kernel(int n, int k, int *graph)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n && j < n)
        {
            graph[i * n + j] = min(graph[i * n + j], graph[i * n + k] + graph[k * n + j]);
        }
    }

    __global__ void floyd(int n, int x_lo, int x_hi, int y_lo, int y_hi, int k, int *graph)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y + y_lo;
        int j = blockIdx.x * blockDim.x + threadIdx.x + x_lo;
        if (i < y_hi && j < x_hi)
        {
            graph[i * n + j] = min(graph[i * n + j], graph[i * n + k] + graph[k * n + j]);
        }
    }
}

// void apsp(int n, /* device */ int *graph)
// {
//     int block_num = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     dim3 thr(32, 32);
//     dim3 blk((BLOCK_SIZE - 1) / 32 + 1, (BLOCK_SIZE - 1) / 32 + 1);
//     for (int p = 0; p < block_num; ++p)
//     {
//         int lo = p * BLOCK_SIZE, hi = (p + 1) * BLOCK_SIZE;
//         hi = (hi > n) ? n : hi;
//         for (int k = lo; k < hi; ++k)
//             floyd<<<blk, thr>>>(n, lo, hi, lo, hi, k, graph);
//         for (int k = lo; k < hi; ++k)
//         {
//             floyd<<<blk, thr>>>(n, lo, hi, 0, lo, k, graph);
//             floyd<<<blk, thr>>>(n, lo, hi, hi, n, k, graph);
//             floyd<<<blk, thr>>>(n, 0, lo, lo, hi, k, graph);
//             floyd<<<blk, thr>>>(n, hi, n, lo, hi, k, graph);
//         }
//         for (int k = lo; k < hi; ++k)
//         {
//             floyd<<<blk, thr>>>(n, 0, lo, 0, lo, k, graph);
//             floyd<<<blk, thr>>>(n, 0, lo, hi, n, k, graph);
//             floyd<<<blk, thr>>>(n, hi, n, 0, lo, k, graph);
//             floyd<<<blk, thr>>>(n, hi, n, hi, n, k, graph);
//         }
//     }
// }

void apsp(int n, /* device */ int *graph)
{
    for (int k = 0; k < n; k++)
    {
        dim3 thr(32, 32);
        dim3 blk((n - 1) / 32 + 1, (n - 1) / 32 + 1);
        kernel<<<blk, thr>>>(n, k, graph);
    }
}